#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] std::span<const TokenId> token_ids() const;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

// One LoRA applied to the loaded DiT for a single image_denoise call: a safetensors path and its
// scale. Entries stack in list order (weight + scale * (B @ A) per entry); the base weights are
// restored after the call.
struct LoraRef {
    std::string path;
    float scale = 1.F;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input) const;

    // Raw token input is retained for parity tools and repeatable performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    // The exact bytes a vocabulary token decodes to, special tokens included. One token may hold a
    // partial UTF-8 sequence. Throws std::out_of_range for an id outside the vocabulary.
    [[nodiscard]] std::string token_bytes(TokenId token) const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously. Delivery intent is fixed before queue
    // membership; terminal-only requests do not publish per-round OutputDelta events,
    // but may use an OutputSink for host-only recovery diagnostics. Destroying
    // an unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputDelivery delivery                                = OutputDelivery::TerminalOnly,
           std::chrono::steady_clock::time_point pending_deadline = {},
           HostInputLease host_input                              = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    // Teacher-forced next-token NLL over a prepared token sequence. Does not sample, decode,
    // or change generate/serve graphs. Requires an idle Engine.
    [[nodiscard]] ScoreResult score(PreparedPrompt prompt, ScoreOptions options = {});

    // One idle admission and execution reservation for the complete batch. No
    // generation can interleave between candidates. Each sequence is evaluated
    // independently; this does not promise shared-prefix computation reuse.
    // At most 16 sequences and 4 * max_context aggregate prepared tokens.
    [[nodiscard]] std::vector<ScoreResult> score_many(std::vector<PreparedPrompt> prompts,
                                                      std::vector<ScoreOptions> options,
                                                      const CancellationView& cancellation = {});

    // Image-pipeline component entry points. Each requires the active target to be the matching
    // component. load_artifact is what swaps the loaded file; the process keeps listening.
    // VAE: latent (host bytes, [1,64,H/16,W/16] BF16) -> image (host bytes, [1,4,H,W] BF16).
    [[nodiscard]] std::vector<std::uint8_t>
    image_decode(const std::uint8_t* latent_bytes, std::size_t latent_bytes_count, int H, int W) const;
    // VAE encode (img2img): image (host bytes, [1,4,H,W] BF16 in [-1,1]) -> latent
    // (host bytes, [1,64,H/16,W/16] BF16, the posterior mode).
    [[nodiscard]] std::vector<std::uint8_t>
    image_vae_encode(const std::uint8_t* image_bytes, std::size_t image_bytes_count, int H, int W) const;
    // DiT: text_emb (host bytes, [1,M,4096] BF16) -> latent (host bytes, [1,N,64] BF16), where
    // image_h x image_w are pixels and N = (image_h/16)*(image_w/16). `noise` is the initial
    // latent in the same layout; when empty it is drawn from `seed`.
    // `loras` empty leaves the loaded DiT weights as they are. Each entry adds
    // `scale` * (B @ A) for this call (omitted scale is 1) and the weights are restored after.
    [[nodiscard]] std::vector<std::uint8_t>
    image_denoise(const std::uint8_t* emb_bytes, std::size_t emb_bytes_count, int M, int image_h,
                  int image_w, const std::vector<std::uint8_t>& noise, std::uint64_t seed,
                  const std::vector<LoraRef>& loras = {}, float strength = 1.0f,
                  const std::vector<std::uint8_t>& cond_bytes = {}, int cond_h = 0, int cond_w = 0,
                  int text_split = 0, int vlm_img_tokens = 0,
                  const std::vector<std::pair<int, int>>& cond_shapes = {},
                  const std::vector<std::pair<int, int>>& vlm_pads = {}) const;
    // Encoder (T2I): prompt text -> hidden (host bytes, [1,seq,4096] BF16).
    // The engine tokenizes the prompt using the Qwen3-VL tokenizer.
    [[nodiscard]] std::vector<std::uint8_t>
    image_encode(const std::string& prompt) const;
    std::vector<std::uint8_t>
    image_encode_v2v(const std::string& prompt, const std::uint8_t* image_bytes,
                     std::size_t image_bytes_count, int* text_split_out = nullptr,
                     int* vlm_img_tokens_out = nullptr) const;
    // Multi-ref (Turbo): up to 10 condition images. `vlm_img_tokens_out` is the token
    // span from the first <|image_pad|> through the last (includes inter-image markers),
    // matching what the DiT drops when inserting packed cond latents.
    std::vector<std::uint8_t>
    image_encode_v2v(const std::string& prompt,
                     const std::vector<std::vector<std::uint8_t>>& images,
                     int* text_split_out = nullptr, int* vlm_img_tokens_out = nullptr,
                     std::vector<std::pair<int, int>>* vlm_pads_out = nullptr) const;

    // Frees the active target, including its GPU weights, then loads `path`. Encode, denoise,
    // and decode afterwards use that file, and only when it is the matching component. A path
    // that is not a `.ninfer` file leaves the current target loaded.
    [[nodiscard]] LoadSummary load_artifact(const std::filesystem::path& path);

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    void reset_memory_peaks() noexcept;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
