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
    // Owns the device allocations that weights point into.
    artifact::MaterializedArtifact backing;
    VaeWeights weights;
};

struct ImageProgram::Impl {
    const LoadedModel* model = nullptr;
    DeviceContext* device     = nullptr;
    VaeWorkspace workspace;
    std::size_t workspace_bytes = 0;

    ~Impl() {
        // a–h are the hipMalloc'd maps. Later letters alias those buffers.
        auto release = [](void* p) noexcept {
            if (p != nullptr) { (void)hipFree(p); }
        };
        release(workspace.a);
        release(workspace.b);
        release(workspace.c);
        release(workspace.d);
        release(workspace.e);
        release(workspace.f);
        release(workspace.g);
        release(workspace.h);
        workspace.a = workspace.b = workspace.c = workspace.d = nullptr;
        workspace.e = workspace.f = workspace.g = workspace.h = nullptr;
    }
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
    // The largest map is the 288-channel upsample at full resolution. Grow the
    // workspace if a later request is bigger than the one that allocated it.
    const std::size_t max_bytes = (std::size_t)288 * H * W * 2;
    if (impl_->workspace_bytes < max_bytes) {
        auto alloc = [&](std::size_t bytes) {
            void* p = nullptr;
            HIP_CHECK(hipMalloc(&p, bytes));
            return p;
        };
        auto& ws = impl_->workspace;
        auto drop = [](hip_bfloat16*& p) {
            HIP_CHECK(hipFree(p));
            p = nullptr;
        };
        drop(ws.a); drop(ws.b); drop(ws.c); drop(ws.d);
        drop(ws.e); drop(ws.f); drop(ws.g); drop(ws.h);
        ws.a = (hip_bfloat16*)alloc(max_bytes);
        ws.b = (hip_bfloat16*)alloc(max_bytes);
        ws.c = (hip_bfloat16*)alloc(max_bytes);
        ws.d = (hip_bfloat16*)alloc(max_bytes);
        ws.e = (hip_bfloat16*)alloc(max_bytes);
        ws.f = (hip_bfloat16*)alloc(max_bytes);
        ws.g = (hip_bfloat16*)alloc(max_bytes);
        ws.h = (hip_bfloat16*)alloc(max_bytes);
        impl_->workspace_bytes = max_bytes;
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
    impl->backing = std::move(materialized);
    return std::unique_ptr<detail::LoadedModel>(new detail::LoadedModel(std::move(impl)));
}

std::unique_ptr<detail::ImageProgram> Package::create_program(const LoadedModel& model, DeviceContext& device) {
    auto impl = std::make_unique<detail::ImageProgram::Impl>();
    impl->model  = &model;
    impl->device = &device;
    return std::unique_ptr<detail::ImageProgram>(new detail::ImageProgram(std::move(impl)));
}

} // namespace ninfer::targets::qwen_image_vae
