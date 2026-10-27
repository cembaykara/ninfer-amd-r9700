#include "ninfer/engine.h"

#include "core/device.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/types.h"
#include "runtime/engine/concurrent_executor.h"
#include "targets/qwen3/impl/frontend/encoded_history_cache.h"
#include "targets/qwen_image_21/impl/lora_apply.h"
#include "targets/qwen_image_21/impl/config.h"
#include "targets/qwen3_vl/impl/vision_preprocess.h"
#include "targets/registry.h"

#include <ninfer/targets/qwen3/prepared_prompt.h>

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace ninfer {
namespace {

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens    = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse         = options.execution.allow_prefix_reuse;
    resolved.execution.capture_context_checkpoint = options.execution.capture_context_checkpoint;
    if (options.output.top_logprobs && *options.output.top_logprobs > kMaximumTopLogprobs) {
        throw std::invalid_argument("top_logprobs exceeds kMaximumTopLogprobs");
    }
    resolved.execution.token_logprobs = options.output.top_logprobs.has_value();
    resolved.stop                     = std::move(options.stop);
    resolved.output                   = options.output;
    return resolved;
}

std::string context_capacity_error(std::uint32_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, double frontend_seconds, SamplingMode mode,
         targets::qwen3::PreparedPrompt prepared)
        : summary(prompt_summary), prepare_seconds(frontend_seconds), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    double prepare_seconds     = 0.0;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    targets::qwen3::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

std::span<const TokenId> PreparedPrompt::token_ids() const {
    if (impl_ == nullptr) { return {}; }
    return targets::qwen3::PreparedPromptAccess::view(impl_->value).token_ids;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    using Executor = runtime::ConcurrentExecutor<targets::Qwen3_8_27BInstance>;

    explicit Impl(EngineOptions engine_options)
        : options(std::move(engine_options)), device(options.device) {
        auto constructed      = targets::construct_target(options, device);
        active                = std::move(constructed.active);
        load                  = std::move(constructed.load);
        sampling_defaults     = constructed.sampling_defaults;
        if (auto* qwen = std::get_if<targets::Qwen3_8_27BInstance>(&*active)) {
            options.mixed_forward = qwen->program->mixed_forward();
            executor              = std::make_unique<Executor>(*qwen, device, options);
        } else {
            // Encoder, DiT, or VAE: no text executor. POST /v1/ninfer/load replaces the file.
        }
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        executor.reset();
        try {
            device.synchronize_all();
            // NOLINTNEXTLINE(bugprone-empty-catch): noexcept teardown has no one to report to
        } catch (...) {}
    }

    EngineOptions options;
    DeviceContext device;
    targets::ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    std::unique_ptr<Executor> executor;
    mutable targets::qwen3::frontend_internal::EncodedHistoryCache host_encode_cache;
    // Serializes load_artifact with image calls and with readers of the active target.
    mutable std::mutex model_mu;
};

Engine::Engine(EngineOptions options) : impl_(std::make_shared<Impl>(std::move(options))) {}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

namespace {
// The active target is a variant (text Qwen3 or the Qwen-Image DiT). These text-specific
// methods only make sense for the Qwen3 instance; the DiT is served via its ImageProgram.
const targets::Qwen3_8_27BInstance& text_target(const targets::ActiveTarget& active) {
    if (active == nullptr) { throw std::logic_error("Engine target is not active"); }
    const auto* qwen = std::get_if<targets::Qwen3_8_27BInstance>(&*active);
    if (qwen == nullptr) { throw std::logic_error("Engine target is not a text model"); }
    return *qwen;
}

const targets::VaeInstance& vae_target(const targets::ActiveTarget& active) {
    if (active == nullptr) { throw std::logic_error("Engine target is not active"); }
    const auto* vae = std::get_if<targets::VaeInstance>(&*active);
    if (vae == nullptr) { throw std::logic_error("Engine target is not the VAE"); }
    return *vae;
}

const targets::DitInstance& dit_target(const targets::ActiveTarget& active) {
    if (active == nullptr) { throw std::logic_error("Engine target is not active"); }
    const auto* dit = std::get_if<targets::DitInstance>(&*active);
    if (dit == nullptr) { throw std::logic_error("Engine target is not the DiT"); }
    return *dit;
}

const targets::EncoderInstance& encoder_target(const targets::ActiveTarget& active) {
    if (active == nullptr) { throw std::logic_error("Engine target is not active"); }
    const auto* enc = std::get_if<targets::EncoderInstance>(&*active);
    if (enc == nullptr) { throw std::logic_error("Engine target is not the encoder"); }
    return *enc;
}
} // namespace

PreparedPrompt Engine::prepare(PromptInput input) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const SamplingMode sampling_mode =
        input.options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    const auto& target = text_target(impl_->active);
    auto prepared = targets::qwen3::EncodedHistoryPrepare::prepare(
        target.loaded->frontend, std::move(input), impl_->host_encode_cache);
    PromptSummary info = prepared.summary();
    if (info.prompt_tokens > target.capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(info.prompt_tokens, target.capacity));
    }
    const double seconds = prepared.prepare_seconds();
    return PreparedPrompt(
        std::make_unique<PreparedPrompt::Impl>(info, seconds, sampling_mode, std::move(prepared)));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const auto& target = text_target(impl_->active);
    auto prepared =
        target.loaded->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity);
    PromptSummary info = prepared.summary();
    if (info.prompt_tokens > target.capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(info.prompt_tokens, target.capacity));
    }
    const double seconds = prepared.prepare_seconds();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
        info, seconds, SamplingMode::Thinking, std::move(prepared)));
}

std::uint32_t Engine::count_tokens(PromptInput input) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const auto& target = text_target(impl_->active);
    return targets::qwen3::EncodedHistoryPrepare::count_tokens(
        target.loaded->frontend, std::move(input), impl_->host_encode_cache);
}

PromptCapabilities Engine::prompt_capabilities() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    // Image targets (encoder / DiT / VAE) have no chat prompt; the server still asks at startup.
    if (impl_->active == nullptr ||
        std::get_if<targets::Qwen3_8_27BInstance>(&*impl_->active) == nullptr) {
        return PromptCapabilities{};
    }
    const auto& target = text_target(impl_->active);
    return target.loaded->frontend.prompt_capabilities();
}

std::string Engine::token_bytes(TokenId token) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const auto& target = text_target(impl_->active);
    return std::string(target.loaded->frontend.token_bytes(token));
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputDelivery delivery,
                                std::chrono::steady_clock::time_point pending_deadline,
                                HostInputLease host_input) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }

    struct HostInputGuard {
        PreparedPrompt* prompt;
        HostInputLease* lease;

        ~HostInputGuard() {
            if (static_cast<bool>(*lease)) {
                prompt->impl_.reset();
                lease->reset();
            }
        }
    } host_input_guard{&prompt, &host_input};

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare_seconds;
    if (resolved_options.execution.capture_context_checkpoint &&
        !context_checkpoint_capture_available(resolved_options.execution.allow_prefix_reuse,
                                              impl_->options.speculative.backend)) {
        throw std::invalid_argument(
            "capture_context_checkpoint requires prefix reuse and a speculative backend");
    }
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;

            GenerationResult wait(OutputSink*, const CancellationView& cancellation) {
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate;

        immediate.result.prompt                  = prompt_summary;
        immediate.result.finish_reason           = FinishReason::OutputLimit;
        immediate.result.timings.prepare_seconds = prepare_seconds;
        immediate.result.timings.total_seconds   = prepare_seconds;
        prompt.impl_.reset();
        host_input.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    if (impl_->executor == nullptr) {
        throw std::logic_error("concurrent Engine executor is unavailable");
    }
    auto submission = impl_->executor->submit(std::move(prompt.impl_->value), prompt_summary,
                                              prepare_seconds, std::move(resolved_options),
                                              delivery, pending_deadline, std::move(host_input));
    return GenerationHandle(
        std::make_unique<GenerationHandle::Impl>(impl_, std::move(submission), resolved_sampling));
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputDelivery delivery =
        sink == nullptr ? OutputDelivery::TerminalOnly : OutputDelivery::Streaming;
    return submit(std::move(prompt), std::move(options), delivery).wait(sink, cancellation);
}

ScoreResult Engine::score(PreparedPrompt prompt, ScoreOptions options) {
    std::vector<PreparedPrompt> prompts;
    prompts.push_back(std::move(prompt));
    auto results = score_many(std::move(prompts), {options});
    return std::move(results.front());
}

std::vector<ScoreResult> Engine::score_many(std::vector<PreparedPrompt> prompts,
                                            std::vector<ScoreOptions> options,
                                            const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompts.empty() || prompts.size() > 16 || options.size() != prompts.size()) {
        throw std::invalid_argument("score_many requires 1..16 prompts with one option per prompt");
    }
    std::uint64_t total_tokens = 0;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        const auto& prompt = prompts[i];
        if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
        const auto count = prompt.impl_->summary.prompt_tokens;
        if (count > impl_->options.max_context) {
            throw RequestError(RequestErrorKind::ContextLengthExceeded,
                               context_capacity_error(count, impl_->options.max_context));
        }
        const ScoreOptions& option = options[i];
        if (count < 2 || (option.schedule == ScoreSchedule::Decode && count < 3)) {
            throw std::invalid_argument("score sequence has too few tokens for its schedule");
        }
        if (option.skip_tokens && *option.skip_tokens > count - 2) {
            throw std::invalid_argument("score skip leaves no teacher-forced target");
        }
        total_tokens += count;
    }
    if (total_tokens > std::uint64_t{impl_->options.max_context} * 4) {
        throw std::invalid_argument("score batch exceeds 4 * max_context prepared tokens");
    }
    if (impl_->executor == nullptr) {
        throw std::logic_error("concurrent Engine executor is unavailable");
    }
    std::vector<targets::qwen3::PreparedPrompt> prepared;
    prepared.reserve(prompts.size());
    for (auto& prompt : prompts) { prepared.push_back(std::move(prompt.impl_->value)); }
    return impl_->executor->score_many(std::move(prepared), options, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    // Image targets run without the text executor; startup and /metrics still read these.
    if (impl_->executor == nullptr) { return MemorySummary{}; }
    return impl_->executor->memory_summary();
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    if (impl_->executor == nullptr) { return RuntimeStats{}; }
    return impl_->executor->runtime_stats();
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    if (impl_->executor != nullptr) { impl_->executor->reset_memory_peaks(); }
}

LoadSummary Engine::load_artifact(const std::filesystem::path& path) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    std::error_code ec;
    const bool is_file = std::filesystem::is_regular_file(path, ec);
    if (path.empty() || path.extension() != ".ninfer" || ec || !is_file) {
        throw std::invalid_argument("artifact path is not a .ninfer file: " + path.string());
    }
    // Copy first so a failure here still leaves the resident target in place. The free
    // below is what lets the next file fit: encoder and DiT do not share the device.
    EngineOptions next = impl_->options;
    next.artifact_path = path;
    impl_->device.bind_to_current_thread();
    impl_->executor.reset();
    impl_->device.synchronize_all();
    impl_->active.reset();
    targets::ConstructedTarget constructed;
    try {
        constructed = targets::construct_target(next, impl_->device);
    } catch (...) {
        impl_->load = LoadSummary{};
        impl_->sampling_defaults = ModelSamplingDefaults{};
        throw;
    }
    impl_->options.artifact_path = path;
    impl_->options.model_id      = constructed.load.model_id;
    impl_->options.weights_id    = constructed.load.weights_id;
    impl_->active                = std::move(constructed.active);
    impl_->load                  = std::move(constructed.load);
    impl_->sampling_defaults     = constructed.sampling_defaults;
    if (auto* qwen = std::get_if<targets::Qwen3_8_27BInstance>(&*impl_->active)) {
        impl_->options.mixed_forward = qwen->program->mixed_forward();
        impl_->executor = std::make_unique<Engine::Impl::Executor>(*qwen, impl_->device, impl_->options);
    }
    return impl_->load;
}

std::vector<std::uint8_t>
Engine::image_decode(const std::uint8_t* latent_bytes, std::size_t latent_bytes_count, int H, int W) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    const auto& target = vae_target(impl_->active);
    void* d_latent = nullptr;
    HIP_CHECK(hipMalloc(&d_latent, latent_bytes_count));
    HIP_CHECK(hipMemcpy(d_latent, latent_bytes, latent_bytes_count, hipMemcpyHostToDevice));
    Tensor latent(d_latent, DType::BF16, {1, 64, H / 16, W / 16});
    const std::size_t out_bytes = (std::size_t)4 * H * W * 2;  // [1,4,H,W] BF16
    void* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, out_bytes));
    Tensor out(d_out, DType::BF16, {1, 4, H, W});
    target.loaded->program->decode(latent, out, impl_->device.stream);
    impl_->device.synchronize();
    std::vector<std::uint8_t> out_host(out_bytes);
    HIP_CHECK(hipMemcpy(out_host.data(), d_out, out_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_latent));
    HIP_CHECK(hipFree(d_out));
    return out_host;
}

std::vector<std::uint8_t>
Engine::image_vae_encode(const std::uint8_t* image_bytes, std::size_t image_bytes_count, int H, int W) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    const auto& target = vae_target(impl_->active);
    void* d_image = nullptr;
    HIP_CHECK(hipMalloc(&d_image, image_bytes_count));
    HIP_CHECK(hipMemcpy(d_image, image_bytes, image_bytes_count, hipMemcpyHostToDevice));
    Tensor image(d_image, DType::BF16, {1, 4, H, W});
    const std::size_t out_bytes = (std::size_t)64 * (H / 16) * (W / 16) * 2;  // [1,64,H/16,W/16] BF16
    void* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, out_bytes));
    Tensor out(d_out, DType::BF16, {1, 64, H / 16, W / 16});
    target.loaded->program->encode(image, out, impl_->device.stream);
    impl_->device.synchronize();
    std::vector<std::uint8_t> out_host(out_bytes);
    HIP_CHECK(hipMemcpy(out_host.data(), d_out, out_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_image));
    HIP_CHECK(hipFree(d_out));
    return out_host;
}

std::vector<std::uint8_t>
Engine::image_denoise(const std::uint8_t* emb_bytes, std::size_t emb_bytes_count, int M, int image_h,
                      int image_w, const std::vector<std::uint8_t>& noise, std::uint64_t seed,
                      const std::vector<LoraRef>& loras, float strength,
                      const std::vector<std::uint8_t>& cond_bytes, int cond_h, int cond_w,
                      int text_split, int vlm_img_tokens,
                      const std::vector<std::pair<int, int>>& cond_shapes,
                      const std::vector<std::pair<int, int>>& vlm_pads) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    const auto& target = dit_target(impl_->active);
    // 8x VAE downsampling and 2x2 packing: one latent token per 16x16 pixels.
    const int grid_h = image_h / 16;
    const int grid_w = image_w / 16;
    const std::size_t latent_bytes = (std::size_t)grid_h * grid_w * 64 * 2;  // {64, N} BF16
    std::vector<std::uint8_t> latent_host = noise;
    if (latent_host.empty()) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        latent_host.resize(latent_bytes);
        // Pack float -> BF16 by hand. The host hip_bfloat16(float) constructor goes
        // through static_cast<__bf16>, which this host toolchain mis-emits as NaN for
        // every value, so every noise sample would be NaN and the whole image NaN.
        // Round-to-nearest-even on the high 16 bits of the IEEE-754 float.
        auto* words = reinterpret_cast<std::uint16_t*>(latent_host.data());
        for (std::size_t i = 0; i < latent_bytes / 2; i++) {
            std::uint32_t bits = 0;
            const float sample = normal(rng);
            std::memcpy(&bits, &sample, sizeof(bits));
            bits += 0x7FFFU + ((bits >> 16) & 1U);
            words[i] = static_cast<std::uint16_t>(bits >> 16);
        }
    } else if (latent_host.size() != latent_bytes) {
        throw std::invalid_argument("image_denoise: noise byte count does not match the image size");
    }
    // Causal edit (model way): pure-noise target + full sigmas. Strength/img2img
    // mix is voided — it fights the fixed cond prefix the DiT was trained with.
    int step_begin = 0;
    (void)strength;
    struct DeviceBuf {
        void* p = nullptr;
        ~DeviceBuf() {
            if (p != nullptr) { (void)hipFree(p); }
        }
    };
    DeviceBuf d_emb;
    DeviceBuf d_latent;
    HIP_CHECK(hipMalloc(&d_emb.p, emb_bytes_count));
    HIP_CHECK(hipMemcpy(d_emb.p, emb_bytes, emb_bytes_count, hipMemcpyHostToDevice));
    Tensor emb(d_emb.p, DType::BF16, {4096, M});
    HIP_CHECK(hipMalloc(&d_latent.p, latent_bytes));
    HIP_CHECK(hipMemcpy(d_latent.p, latent_host.data(), latent_bytes, hipMemcpyHostToDevice));
    Tensor out_latent(d_latent.p, DType::BF16, {64, grid_h * grid_w});
    // Condition latent(s); empty for text-to-image. cond_shapes lists each pack grid; when empty,
    // a single (cond_h, cond_w) is used.
    std::vector<int> shape_hs;
    std::vector<int> shape_ws;
    if (!cond_shapes.empty()) {
        for (const auto& hw : cond_shapes) {
            shape_hs.push_back(hw.first);
            shape_ws.push_back(hw.second);
        }
    } else if (cond_h > 0 && cond_w > 0) {
        shape_hs.push_back(cond_h);
        shape_ws.push_back(cond_w);
    }
    int Nc = 0;
    for (std::size_t i = 0; i < shape_hs.size(); ++i) { Nc += shape_hs[i] * shape_ws[i]; }
    DeviceBuf d_cond;
    Tensor cond_latent;
    if (Nc > 0) {
        const std::size_t cond_bytes_expected = (std::size_t)Nc * 64 * 2;
        if (cond_bytes.size() != cond_bytes_expected) {
            throw std::invalid_argument("image_denoise: condition byte count does not match the grid");
        }
        HIP_CHECK(hipMalloc(&d_cond.p, cond_bytes_expected));
        HIP_CHECK(hipMemcpy(d_cond.p, cond_bytes.data(), cond_bytes_expected, hipMemcpyHostToDevice));
        cond_latent = Tensor(d_cond.p, DType::BF16, {64, Nc});
    }
    // T2I: cond_latent stays default-constructed (the forward skips it when Nc == 0).
    // No LoRA: the weights loaded from the DiT file stay as they are. Each LoRA adds
    // scale * (B @ A) for this call (scale defaults to 1) and the weights are restored after.
    std::vector<int> pad_starts;
    std::vector<int> pad_counts;
    if (!vlm_pads.empty()) {
        pad_starts.reserve(vlm_pads.size());
        pad_counts.reserve(vlm_pads.size());
        for (const auto& hw : vlm_pads) {
            pad_starts.push_back(hw.first);
            pad_counts.push_back(hw.second);
        }
    } else if (!shape_hs.empty()) {
        // Legacy single-span: one pad run covering text_split..+vlm_img_tokens.
        pad_starts.push_back(text_split);
        pad_counts.push_back(vlm_img_tokens > 0 ? vlm_img_tokens : 0);
        if (pad_counts[0] <= 0 && !shape_hs.empty()) {
            throw std::invalid_argument("image_denoise: vlm_img_tokens required without vlm_pads");
        }
    }
    if (!shape_hs.empty() && pad_starts.size() != shape_hs.size()) {
        // Multi-ref without per-image pads: synthesize contiguous fake runs (old glued behaviour)
        // only when a single pad span was provided for multiple shapes — reject.
        if (vlm_pads.empty() && shape_hs.size() > 1) {
            throw std::invalid_argument(
                "image_denoise: multi-ref requires vlm_pads from encoder (one pad run per image)");
        }
    }
    const auto run_denoise = [&]() {
        if (shape_hs.empty()) {
            target.loaded->program->denoise(emb, text_split, vlm_img_tokens, 0, 0, grid_h, grid_w,
                                            cond_latent, out_latent, impl_->device.stream, step_begin);
            return;
        }
        if (pad_starts.size() != shape_hs.size()) {
            throw std::invalid_argument("image_denoise: vlm_pads size must match cond_shapes");
        }
        target.loaded->program->denoise(emb, pad_starts.data(), pad_counts.data(), shape_hs.data(),
                                        shape_ws.data(), static_cast<int>(shape_hs.size()), grid_h,
                                        grid_w, cond_latent, out_latent, impl_->device.stream,
                                        step_begin);
    };
    if (loras.empty()) {
        run_denoise();
    } else {
        std::vector<targets::qwen_image_21::detail::LoraRef> refs;
        refs.reserve(loras.size());
        for (const LoraRef& ref : loras) {
            refs.push_back({std::filesystem::path(ref.path), ref.scale});
        }
        targets::qwen_image_21::detail::denoise_with_lora(
            target.loaded->model->weights(), impl_->options.artifact_path, refs,
            impl_->device.stream, run_denoise);
    }
    impl_->device.synchronize();
    HIP_CHECK(hipMemcpy(latent_host.data(), d_latent.p, latent_bytes, hipMemcpyDeviceToHost));
    return latent_host;
}

std::vector<std::uint8_t>
Engine::image_encode(const std::string& prompt) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    const auto& target = encoder_target(impl_->active);
    // Tokenize the prompt using the Qwen3-VL tokenizer (loaded from the Turbo checkpoint).
    const auto& tokenizer = target.loaded->model->tokenizer();
    // The QwenImage21Pipeline T2I template. Its system turn is 14 tokens, which diffusers
    // drops from the hidden states (`_drop_idx`).
    constexpr int kDropTokens = 14;
    const std::string text =
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int> ids = tokenizer.encode(text);
    const int seq = static_cast<int>(ids.size());
    if (seq <= kDropTokens) { throw std::runtime_error("prompt tokenized to too few tokens"); }
    const std::size_t ids_bytes = (std::size_t)seq * 4;
    void* d_ids = nullptr;
    HIP_CHECK(hipMalloc(&d_ids, ids_bytes));
    HIP_CHECK(hipMemcpy(d_ids, ids.data(), ids_bytes, hipMemcpyHostToDevice));
    Tensor ids_tensor(d_ids, DType::I32, {seq});

    const std::size_t row_bytes    = (std::size_t)4096 * 2;
    const std::size_t hidden_bytes = (std::size_t)seq * row_bytes;
    void* d_hidden = nullptr;
    HIP_CHECK(hipMalloc(&d_hidden, hidden_bytes));
    Tensor out_hidden(d_hidden, DType::BF16, {4096, seq});

    target.loaded->program->embed(ids_tensor, out_hidden, impl_->device.stream);
    impl_->device.synchronize();

    const std::size_t kept_bytes = (std::size_t)(seq - kDropTokens) * row_bytes;
    std::vector<std::uint8_t> hidden_host(kept_bytes);
    HIP_CHECK(hipMemcpy(hidden_host.data(),
                        static_cast<const std::uint8_t*>(d_hidden) + kDropTokens * row_bytes,
                        kept_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_ids));
    HIP_CHECK(hipFree(d_hidden));
    return hidden_host;
}

std::vector<std::uint8_t>
Engine::image_encode_v2v(const std::string& prompt, const std::uint8_t* image_bytes,
                         std::size_t image_bytes_count, int* text_split_out,
                         int* vlm_img_tokens_out) const {
    std::vector<std::vector<std::uint8_t>> images(1);
    images[0].assign(image_bytes, image_bytes + image_bytes_count);
    return image_encode_v2v(prompt, images, text_split_out, vlm_img_tokens_out);
}

// v2v multi-ref: preprocess each image, build HF-style <imageN> pad blocks, ViT with
// per-image cu_seqlens, inject at each pad run, drop 14 system tokens.
std::vector<std::uint8_t>
Engine::image_encode_v2v(const std::string& prompt,
                         const std::vector<std::vector<std::uint8_t>>& images,
                         int* text_split_out, int* vlm_img_tokens_out,
                         std::vector<std::pair<int, int>>* vlm_pads_out) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (images.empty() || images.size() > 10) {
        throw std::invalid_argument("v2v: need 1..10 condition images");
    }
    std::lock_guard<std::mutex> lock(impl_->model_mu);
    const auto& target = encoder_target(impl_->active);
    const auto& tokenizer = target.loaded->model->tokenizer();
    constexpr int kDropTokens = 14;
    constexpr int kPatchFeatures = 1536;

    const int n = static_cast<int>(images.size());
    std::vector<targets::qwen3_vl::detail::V2VImagePrep> preps;
    preps.reserve(static_cast<std::size_t>(n));
    std::vector<int> grid_hs(n), grid_ws(n), visual_counts(n), cu_seqlens(n + 1);
    cu_seqlens[0] = 0;
    int P_total = 0;
    int V_total = 0;
    std::vector<float> patches_all;
    std::vector<int> position_ids_all;
    std::vector<int> pos_indices_all;
    std::vector<float> pos_weights_all;
    for (int i = 0; i < n; ++i) {
        if (images[static_cast<std::size_t>(i)].empty()) {
            throw std::invalid_argument("v2v: empty condition image");
        }
        auto prep = targets::qwen3_vl::detail::preprocess_v2v_image(
            images[static_cast<std::size_t>(i)].data(), images[static_cast<std::size_t>(i)].size());
        grid_hs[i] = prep.grid_h;
        grid_ws[i] = prep.grid_w;
        visual_counts[i] = prep.tokens;
        P_total += prep.patches_count;
        V_total += prep.tokens;
        cu_seqlens[i + 1] = P_total;
        patches_all.insert(patches_all.end(), prep.patches.begin(), prep.patches.end());
        position_ids_all.insert(position_ids_all.end(), prep.position_ids.begin(),
                                prep.position_ids.end());
        pos_indices_all.insert(pos_indices_all.end(), prep.pos_indices.begin(),
                               prep.pos_indices.end());
        pos_weights_all.insert(pos_weights_all.end(), prep.pos_weights.begin(),
                               prep.pos_weights.end());
        preps.push_back(std::move(prep));
    }

    const int image_pad_id = tokenizer.encode("<|image_pad|>")[0];
    std::string text =
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        "<|im_start|>user\n";
    for (int i = 0; i < n; ++i) {
        if (i > 0) { text += " "; }
        text += "<image" + std::to_string(i + 1) + "><|vision_start|>";
        for (int t = 0; t < visual_counts[static_cast<std::size_t>(i)]; ++t) {
            text += "<|image_pad|>";
        }
        text += "<|vision_end|>";
    }
    text += prompt + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int> ids = tokenizer.encode(text);
    const int seq = static_cast<int>(ids.size());
    if (seq <= kDropTokens) { throw std::runtime_error("v2v prompt tokenized to too few tokens"); }

    std::vector<int> visual_starts(n, -1);
    {
        int img = 0;
        int run = 0;
        for (int i = 0; i < seq && img < n; ++i) {
            if (ids[static_cast<std::size_t>(i)] == image_pad_id) {
                if (run == 0) { visual_starts[static_cast<std::size_t>(img)] = i; }
                ++run;
                if (run == visual_counts[static_cast<std::size_t>(img)]) {
                    ++img;
                    run = 0;
                }
            } else if (run != 0) {
                throw std::runtime_error("v2v: broken <|image_pad|> run");
            }
        }
        if (img != n) { throw std::runtime_error("v2v: expected image_pad runs not found"); }
    }
    const int first_pad = visual_starts[0];
    const int last_pad_end =
        visual_starts[static_cast<std::size_t>(n - 1)] + visual_counts[static_cast<std::size_t>(n - 1)];
    const int vlm_span = last_pad_end - first_pad;

    const std::size_t ids_bytes = (std::size_t)seq * 4;
    void* d_ids = nullptr;
    HIP_CHECK(hipMalloc(&d_ids, ids_bytes));
    HIP_CHECK(hipMemcpy(d_ids, ids.data(), ids_bytes, hipMemcpyHostToDevice));
    Tensor ids_tensor(d_ids, DType::I32, {seq});

    const std::size_t patches_bytes = (std::size_t)P_total * kPatchFeatures * sizeof(float);
    void* d_patches = nullptr;
    HIP_CHECK(hipMalloc(&d_patches, patches_bytes));
    HIP_CHECK(hipMemcpy(d_patches, patches_all.data(), patches_bytes, hipMemcpyHostToDevice));

    const std::size_t row_bytes    = (std::size_t)4096 * 2;
    const std::size_t hidden_bytes = (std::size_t)seq * row_bytes;
    void* d_hidden = nullptr;
    HIP_CHECK(hipMalloc(&d_hidden, hidden_bytes));
    Tensor out_hidden(d_hidden, DType::BF16, {4096, seq});

    target.loaded->program->embed_v2v_multi(
        ids_tensor, static_cast<const float*>(d_patches), P_total, grid_hs.data(), grid_ws.data(),
        cu_seqlens.data(), n, position_ids_all.data(), pos_indices_all.data(),
        pos_weights_all.data(), visual_starts.data(), visual_counts.data(), out_hidden,
        impl_->device.stream);
    impl_->device.synchronize();

    const std::size_t kept_bytes = (std::size_t)(seq - kDropTokens) * row_bytes;
    std::vector<std::uint8_t> hidden_host(kept_bytes);
    HIP_CHECK(hipMemcpy(hidden_host.data(),
                        static_cast<const std::uint8_t*>(d_hidden) + kDropTokens * row_bytes,
                        kept_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_ids));
    HIP_CHECK(hipFree(d_patches));
    HIP_CHECK(hipFree(d_hidden));
    if (text_split_out != nullptr) { *text_split_out = first_pad - kDropTokens; }
    if (vlm_img_tokens_out != nullptr) { *vlm_img_tokens_out = vlm_span; }
    if (vlm_pads_out != nullptr) {
        vlm_pads_out->clear();
        vlm_pads_out->reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            vlm_pads_out->emplace_back(visual_starts[static_cast<std::size_t>(i)] - kDropTokens,
                                       visual_counts[static_cast<std::size_t>(i)]);
        }
    }
    (void)V_total;
    (void)preps;
    return hidden_host;
}

} // namespace ninfer
