#include "ninfer/targets/qwen3_vl/package.h"

#include "targets/qwen3/impl/frontend/tokenizer.h"
#include "targets/qwen3_vl/impl/encoder_forward.h"
#include "targets/qwen3_vl/impl/load/bindings.h"
#include "targets/qwen3_vl/impl/v2v_forward.h"
#include "targets/qwen3_vl/impl/vision_forward.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/device.h"

#include <hip/hip_runtime.h>

#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace ninfer::targets::qwen3_vl {
namespace detail {

struct LoadPlan::Impl {
    artifact::MaterializationPlan plan;
    EncoderWeightHandles handles;
    VisionWeightHandles vision_handles;
};

struct LoadedModel::Impl {
    // Owns the device allocations that weights / vision_weights point into.
    artifact::MaterializedArtifact backing;
    EncoderWeights weights;
    VisionWeights vision_weights;
    // The Qwen3-VL tokenizer (loaded from the Turbo checkpoint's processor/ dir on disk).
    // The std::string members own the bytes so the Tokenizer's string_views stay valid.
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string generation_config_json;
    std::unique_ptr<qwen3::frontend_internal::Tokenizer> tokenizer;
};

// Load the Qwen3-VL tokenizer from the Turbo checkpoint's processor/ dir on disk.
void load_vl_tokenizer(LoadedModel::Impl& impl, const std::string& processor_dir) {
    auto read_file = [](const std::string& path) -> std::string {
        std::ifstream f(path, std::ios::binary);
        if (!f) { throw std::runtime_error("failed to open tokenizer file: " + path); }
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };
    impl.tokenizer_json         = read_file(processor_dir + "/tokenizer.json");
    impl.tokenizer_config_json  = read_file(processor_dir + "/tokenizer_config.json");
    impl.generation_config_json = read_file(processor_dir + "/generation_config.json");
    qwen3::frontend_internal::TokenizerResources resources{
        .tokenizer_json         = impl.tokenizer_json,
        .tokenizer_config_json  = impl.tokenizer_config_json,
        .generation_config_json = impl.generation_config_json,
    };
    impl.tokenizer = std::make_unique<qwen3::frontend_internal::Tokenizer>(resources);
}

void release_device(void* p) noexcept {
    if (p != nullptr) { (void)hipFree(p); }
}

struct EncoderProgram::Impl {
    const LoadedModel* model = nullptr;
    DeviceContext* device    = nullptr;
    EncoderWorkspace workspace;
    // v2v: ViT workspace + output buffers (sized for the first request's patch count)
    VisionWorkspace vision_ws;
    hip_bfloat16* visual_out = nullptr;   // [4096, V]
    hip_bfloat16* deepstack_out = nullptr; // [3][4096, V]
    int vision_p = 0;

    ~Impl() {
        release_device(workspace.x);
        release_device(workspace.xn);
        release_device(workspace.q);
        release_device(workspace.k);
        release_device(workspace.v);
        release_device(workspace.qn);
        release_device(workspace.kn);
        release_device(workspace.attn);
        release_device(workspace.gate);
        release_device(workspace.up);
        release_device(workspace.mlp);
        release_device(workspace.positions);
        release_device(vision_ws.x);
        release_device(vision_ws.h);
        release_device(vision_ws.qkv);
        release_device(vision_ws.attended);
        release_device(vision_ws.projected);
        release_device(vision_ws.up);
        release_device(vision_ws.down);
        release_device(vision_ws.merged);
        release_device(vision_ws.mhidden);
        release_device(vision_ws.position_ids);
        release_device(vision_ws.cu_seqlens);
        release_device(vision_ws.pos_indices);
        release_device(vision_ws.pos_weights);
        release_device(visual_out);
        release_device(deepstack_out);
    }
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan() = default;
const artifact::MaterializationPlan& LoadPlan::materialization() const { return impl_->plan; }

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;
const EncoderWeights& LoadedModel::weights() const { return impl_->weights; }
const VisionWeights& LoadedModel::vision_weights() const { return impl_->vision_weights; }
const qwen3::frontend_internal::Tokenizer& LoadedModel::tokenizer() const { return *impl_->tokenizer; }

EncoderProgram::EncoderProgram(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
EncoderProgram::~EncoderProgram() = default;

// Grow the text-stack workspace to  tokens. Both embed and embed_v2v size it this way,
// so a T2I and a v2v call can alternate without one leaving the other undersized.
void ensure_text_workspace(EncoderWorkspace& ws, int seq) {
    if (ws.capacity >= seq) { return; }
    const int H        = EncoderConfig::hidden;
    const int heads    = EncoderConfig::heads;
    const int kv_heads = EncoderConfig::kv_heads;
    const int head_dim = EncoderConfig::head_dim;
    const int inter    = EncoderConfig::intermediate;
    auto grow = [&](auto*& p, std::size_t bytes) {
        if (p != nullptr) {
            HIP_CHECK(hipFree(p));
            p = nullptr;
        }
        void* raw = nullptr;
        HIP_CHECK(hipMalloc(&raw, bytes));
        p = static_cast<std::remove_reference_t<decltype(p)>>(raw);
    };
    const std::size_t n = (std::size_t)seq * 2;
    grow(ws.x, n * H);
    grow(ws.xn, n * H);
    grow(ws.q, n * heads * head_dim);
    grow(ws.k, n * kv_heads * head_dim);
    grow(ws.v, n * kv_heads * head_dim);
    grow(ws.qn, n * heads * head_dim);
    grow(ws.kn, n * kv_heads * head_dim);
    grow(ws.attn, n * heads * head_dim);
    grow(ws.gate, n * inter);
    grow(ws.up, n * inter);
    grow(ws.mlp, n * inter);
    grow(ws.positions, (std::size_t)seq * 4);
    std::vector<int> hpos(seq);
    for (int i = 0; i < seq; i++) { hpos[i] = i; }
    HIP_CHECK(hipMemcpy(ws.positions, hpos.data(), (std::size_t)seq * 4, hipMemcpyHostToDevice));
    ws.capacity = seq;
}

void EncoderProgram::embed(const Tensor& ids, Tensor& out_hidden, hipStream_t stream) const {
    const int seq = ids.ne[0];
    ensure_text_workspace(impl_->workspace, seq);
    encoder_forward(impl_->model->weights(), ids, out_hidden, impl_->workspace, seq, stream);
}

void EncoderProgram::embed_v2v(const Tensor& ids, const float* patches, int patches_count,
                               const int* position_ids, const int* pos_indices,
                               const float* pos_weights, int visual_start, Tensor& out_hidden,
                               hipStream_t stream) const {
    const int seq = ids.ne[0];
    const int P = patches_count;
    const int V = P / VisionConfig::merge_unit;
    const int H = EncoderConfig::hidden;
    const int VH = VisionConfig::hidden;

    // text stack workspace (shared with embed; grown to fit the longer of the two)
    ensure_text_workspace(impl_->workspace, seq);

    // ViT workspace (sized for P patches; freed before regrowing so mixed sizes do not leak)
    if (impl_->vision_p != P) {
        auto alloc = [&](std::size_t bytes) {
            void* p = nullptr;
            HIP_CHECK(hipMalloc(&p, bytes));
            return p;
        };
        auto& ws = impl_->vision_ws;
        release_device(ws.x);
        release_device(ws.patch_in);
        release_device(ws.h);
        release_device(ws.qkv);
        release_device(ws.attended);
        release_device(ws.projected);
        release_device(ws.up);
        release_device(ws.down);
        release_device(ws.merged);
        release_device(ws.mhidden);
        release_device(ws.position_ids);
        release_device(ws.cu_seqlens);
        release_device(ws.pos_indices);
        release_device(ws.pos_weights);
        release_device(impl_->visual_out);
        release_device(impl_->deepstack_out);
        ws.patch_in  = (hip_bfloat16*)alloc((std::size_t)P * 1536 * 2);
        ws.x         = (hip_bfloat16*)alloc((std::size_t)P * VH * 2);
        ws.h         = (hip_bfloat16*)alloc((std::size_t)P * VH * 2);
        ws.qkv       = (hip_bfloat16*)alloc((std::size_t)P * 3 * VH * 2);
        ws.attended  = (hip_bfloat16*)alloc((std::size_t)P * VH * 2);
        ws.projected = (hip_bfloat16*)alloc((std::size_t)P * VH * 2);
        ws.up        = (hip_bfloat16*)alloc((std::size_t)P * VisionConfig::intermediate * 2);
        ws.down      = (hip_bfloat16*)alloc((std::size_t)P * VH * 2);
        ws.merged    = (hip_bfloat16*)alloc((std::size_t)V * VisionConfig::merger_hidden * 2);
        ws.mhidden   = (hip_bfloat16*)alloc((std::size_t)V * VisionConfig::merger_hidden * 2);
        ws.position_ids = (int*)alloc((std::size_t)P * 2 * 4);
        ws.cu_seqlens   = (int*)alloc(2 * 4);
        ws.pos_indices  = (int*)alloc((std::size_t)P * 4 * 4);
        ws.pos_weights  = (float*)alloc((std::size_t)P * 4 * 4);
        impl_->visual_out    = (hip_bfloat16*)alloc((std::size_t)V * H * 2);
        impl_->deepstack_out = (hip_bfloat16*)alloc((std::size_t)3 * V * H * 2);
        impl_->vision_p = P;
    }

    // 1. ViT: patches -> merged [4096, V] + 3 deepstack [4096, V]
    const int cu_seqlens[2] = {0, P};  // one image is one attention segment
    vision_forward(impl_->model->vision_weights(), patches, P, 1,
                   position_ids, cu_seqlens, pos_indices, pos_weights,
                   impl_->visual_out, impl_->deepstack_out, impl_->vision_ws, stream);

    // 2. text stack with deepstack injection
    const hip_bfloat16* ds[3] = {
        impl_->deepstack_out,
        impl_->deepstack_out + (std::size_t)V * H,
        impl_->deepstack_out + (std::size_t)2 * V * H,
    };
    encoder_forward_v2v(impl_->model->weights(), ids, out_hidden, impl_->workspace, seq,
                        impl_->visual_out, visual_start, V, ds, stream);
}

} // namespace detail

detail::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    (void)identity;
    return detail::WeightsProfile::R9700Bf16;
}

detail::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                    detail::WeightsProfile weights_profile) {
    (void)options;
    (void)weights_profile;
    auto impl = std::make_unique<detail::LoadPlan::Impl>();
    impl->handles = detail::plan_encoder_weights(binder);
    impl->vision_handles = detail::plan_vision_weights(binder);
    impl->plan    = binder.finish();
    return LoadPlan(std::move(impl));
}

std::unique_ptr<detail::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    auto impl = std::make_unique<detail::LoadedModel::Impl>();
    detail::bind_encoder_weights(impl->weights, plan.impl_->handles, materialized);
    detail::bind_vision_weights(impl->vision_weights, plan.impl_->vision_handles, materialized);
    // The Qwen3-VL tokenizer lives in the Turbo checkpoint's processor/ dir (on disk).
    detail::load_vl_tokenizer(*impl, "/home/tengri/mergen-api/models/hf/Qwen/Qwen-Image-2.1-Turbo/processor");
    impl->backing = std::move(materialized);
    return std::unique_ptr<detail::LoadedModel>(new detail::LoadedModel(std::move(impl)));
}

std::unique_ptr<detail::EncoderProgram> Package::create_program(const LoadedModel& model, DeviceContext& device) {
    auto impl = std::make_unique<detail::EncoderProgram::Impl>();
    impl->model  = &model;
    impl->device = &device;
    return std::unique_ptr<detail::EncoderProgram>(new detail::EncoderProgram(std::move(impl)));
}

} // namespace ninfer::targets::qwen3_vl
