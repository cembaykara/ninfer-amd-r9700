#include "ninfer/targets/qwen_image_vae/package.h"

#include "targets/qwen_image_vae/impl/load/bindings.h"
#include "targets/qwen_image_vae/impl/vae_forward.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/device.h"

#include <hip/hip_runtime.h>

#include <memory>

namespace ninfer::targets::qwen_image_vae {
namespace detail {

struct LoadPlan::Impl {
    artifact::MaterializationPlan plan;
    VaeWeightHandles handles;
};

struct LoadedModel::Impl {
    VaeWeights weights;
};

struct ImageProgram::Impl {
    const LoadedModel* model = nullptr;
    DeviceContext* device     = nullptr;
    VaeWorkspace workspace;
    bool workspace_ready = false;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan() = default;
const artifact::MaterializationPlan& LoadPlan::materialization() const { return impl_->plan; }

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;
const VaeWeights& LoadedModel::weights() const { return impl_->weights; }

ImageProgram::ImageProgram(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
ImageProgram::~ImageProgram() = default;

void ImageProgram::decode(const Tensor& latent, Tensor& out_image, hipStream_t stream) const {
    // latent: [1, 64, H/16, W/16]; out_image: [1, 4, H, W]
    const int h0 = latent.ne[2], w0 = latent.ne[3];
    const int H = h0 * 16, W = w0 * 16;
    if (!impl_->workspace_ready) {
        // Allocate the workspace, sized for this resolution. The largest intermediate is
        // the up_block-4 / head feature map [144, H, W]; every buffer is reused, so size
        // each to that max. (One-time cost; the VAE decode is not in the denoise loop.)
        auto alloc = [&](std::size_t bytes) {
            void* p = nullptr;
            HIP_CHECK(hipMalloc(&p, bytes));
            return p;
        };
        const std::size_t max_bytes = (std::size_t)144 * H * W * 2;  // [144, H, W] BF16
        auto& ws = impl_->workspace;
        ws.a = (hip_bfloat16*)alloc(max_bytes);
        ws.b = (hip_bfloat16*)alloc(max_bytes);
        ws.c = (hip_bfloat16*)alloc(max_bytes);
        ws.d = (hip_bfloat16*)alloc(max_bytes);
        ws.e = (hip_bfloat16*)alloc(max_bytes);
        ws.f = (hip_bfloat16*)alloc(max_bytes);
        ws.g = (hip_bfloat16*)alloc(max_bytes);
        // (h..z are unused by the current forward; leave null)
        impl_->workspace_ready = true;
    }
    vae_decode(impl_->model->weights(), latent, out_image, impl_->workspace, stream);
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
    impl->handles = plan_vae_weights(binder);
    impl->plan    = binder.finish();
    return LoadPlan(std::move(impl));
}

std::unique_ptr<detail::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    auto impl = std::make_unique<detail::LoadedModel::Impl>();
    bind_vae_weights(impl->weights, plan.impl_->handles, materialized);
    return std::unique_ptr<detail::LoadedModel>(new detail::LoadedModel(std::move(impl)));
}

std::unique_ptr<detail::ImageProgram> Package::create_program(const LoadedModel& model, DeviceContext& device) {
    auto impl = std::make_unique<detail::ImageProgram::Impl>();
    impl->model  = &model;
    impl->device = &device;
    return std::unique_ptr<detail::ImageProgram>(new detail::ImageProgram(std::move(impl)));
}

} // namespace ninfer::targets::qwen_image_vae
