#include "ninfer/targets/qwen_image_21/package.h"

#include "targets/qwen_image_21/impl/dit_forward.h"
#include "targets/qwen_image_21/impl/load/bindings.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/device.h"

#include <hip/hip_runtime.h>

#include <memory>

namespace ninfer::targets::qwen_image_21 {
namespace detail {

struct LoadPlan::Impl {
    artifact::MaterializationPlan plan;
    DitWeightHandles handles;
};

struct LoadedModel::Impl {
    DitWeights weights;
};

struct ImageProgram::Impl {
    const LoadedModel* model = nullptr;
    DeviceContext* device     = nullptr;
    DitWorkspace workspace;
    bool workspace_ready = false;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan() = default;
const artifact::MaterializationPlan& LoadPlan::materialization() const { return impl_->plan; }

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;
const DitWeights& LoadedModel::weights() const { return impl_->weights; }

ImageProgram::ImageProgram(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
ImageProgram::~ImageProgram() = default;

void ImageProgram::denoise(const Tensor& text_emb, int image_h, int image_w, Tensor& out_latent,
                           hipStream_t stream) const {
    const int M = text_emb.ne[1];
    const int N = image_h * image_w;
    if (!impl_->workspace_ready) {
        // allocate the workspace (sized for this M, N); a real target would size for the max
        auto& ws = impl_->workspace;
        const int H = DitConfig::hidden;
        const int C = DitConfig::in_channels;
        const int S = M + N;
        auto alloc = [&](std::size_t bytes) {
            void* p = nullptr;
            HIP_CHECK(hipMalloc(&p, bytes));
            return p;
        };
        ws.img    = (hip_bfloat16*)alloc(N * H * 2);
        ws.txt    = (hip_bfloat16*)alloc(M * H * 2);
        ws.hs     = (hip_bfloat16*)alloc(S * H * 2);
        ws.hs2    = (hip_bfloat16*)alloc(S * H * 2);
        ws.temb   = (hip_bfloat16*)alloc(H * 2);
        ws.temb_mid = (hip_bfloat16*)alloc(H * 2);
        ws.mod    = (hip_bfloat16*)alloc(4 * H * 2);
        ws.q      = (hip_bfloat16*)alloc(S * H * 2);
        ws.k      = (hip_bfloat16*)alloc(S * H * 2);
        ws.v      = (hip_bfloat16*)alloc(S * H * 2);
        ws.attn   = (hip_bfloat16*)alloc(S * H * 2);
        ws.x1     = (hip_bfloat16*)alloc(S * H * 2);
        ws.x2     = (hip_bfloat16*)alloc(S * H * 2);
        ws.mlp    = (hip_bfloat16*)alloc(S * 12288 * 2);
        ws.out    = (hip_bfloat16*)alloc(S * C * 2);
        ws.freqs  = (float*)alloc(S * 128 * 4);
        ws.latent   = (hip_bfloat16*)alloc(N * C * 2);
        ws.velocity = (hip_bfloat16*)alloc(N * C * 2);
        impl_->workspace_ready = true;
    }
    dit_denoise(impl_->model->weights(), text_emb, M, N, out_latent, impl_->workspace, stream);
}

} // namespace detail

WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    (void)identity;
    return WeightsProfile::R9700Bf16;
}

LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                            WeightsProfile weights_profile) {
    (void)options;
    (void)weights_profile;
    auto impl = std::make_unique<LoadPlan::Impl>();
    impl->handles = plan_dit_weights(binder);
    impl->plan    = binder.finish();
    return LoadPlan(std::move(impl));
}

std::unique_ptr<LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    auto impl = std::make_unique<LoadedModel::Impl>();
    bind_dit_weights(impl->weights, plan.impl_->handles, materialized);
    return std::make_unique<LoadedModel>(std::move(impl));
}

std::unique_ptr<ImageProgram> Package::create_program(const LoadedModel& model, DeviceContext& device) {
    auto impl = std::make_unique<ImageProgram::Impl>();
    impl->model  = &model;
    impl->device = &device;
    return std::unique_ptr<ImageProgram>(new ImageProgram(std::move(impl)));
}

} // namespace ninfer::targets::qwen_image_21
