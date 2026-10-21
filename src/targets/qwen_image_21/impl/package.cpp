#include "ninfer/targets/qwen_image_21/package.h"

#include "targets/qwen_image_21/impl/dit_forward.h"
#include "targets/qwen_image_21/impl/load/bindings.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/device.h"

#include <hip/hip_runtime.h>

#include <initializer_list>
#include <memory>
#include <type_traits>

namespace ninfer::targets::qwen_image_21 {
namespace detail {

struct LoadPlan::Impl {
    artifact::MaterializationPlan plan;
    DitWeightHandles handles;
};

struct LoadedModel::Impl {
    // Owns the device allocations that weights point into.
    artifact::MaterializedArtifact backing;
    DitWeights weights;
};

struct ImageProgram::Impl {
    const LoadedModel* model = nullptr;
    DeviceContext* device     = nullptr;
    DitWorkspace workspace;

    ~Impl() {
        auto release = [](void* p) noexcept {
            if (p != nullptr) { (void)hipFree(p); }
        };
        release(workspace.hs);
        release(workspace.x1);
        release(workspace.x2);
        release(workspace.q);
        release(workspace.k);
        release(workspace.v);
        release(workspace.qn);
        release(workspace.kn);
        release(workspace.attn);
        release(workspace.gate);
        release(workspace.up);
        release(workspace.act);
        release(workspace.txt);
        release(workspace.freqs);
        release(workspace.velocity);
        release(workspace.t_sin);
        release(workspace.t_mid);
        release(workspace.temb);
        release(workspace.temb_silu);
        release(workspace.mod);
        release(workspace.out_scale);
        workspace = {};
    }
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

void ImageProgram::denoise(const Tensor& text_emb, int grid_h, int grid_w, Tensor& latent,
                           hipStream_t stream) const {
    auto& ws    = impl_->workspace;
    const int S = static_cast<int>(text_emb.ne[1]) + grid_h * grid_w;
    if (ws.capacity < S) {
        HIP_CHECK(hipStreamSynchronize(stream));
        constexpr std::size_t H = DitConfig::hidden;
        constexpr std::size_t I = DitConfig::intermediate;
        const std::size_t s     = static_cast<std::size_t>(S);
        auto grow = [](auto*& p, std::size_t bytes) {
            if (p != nullptr) {
                HIP_CHECK(hipFree(p));
                p = nullptr;
            }
            void* raw = nullptr;
            HIP_CHECK(hipMalloc(&raw, bytes));
            p = static_cast<std::remove_reference_t<decltype(p)>>(raw);
        };
        for (hip_bfloat16** p : {&ws.hs, &ws.x1, &ws.x2, &ws.q, &ws.k, &ws.v, &ws.qn, &ws.kn, &ws.attn, &ws.txt}) {
            grow(*p, s * H * 2);
        }
        for (hip_bfloat16** p : {&ws.gate, &ws.up, &ws.act}) grow(*p, s * I * 2);
        grow(ws.freqs, s * 128 * sizeof(float));
        grow(ws.velocity, s * DitConfig::in_channels * 2);
        if (ws.mod == nullptr) {
            grow(ws.t_sin, 256 * 2 * 2);
            for (hip_bfloat16** p : {&ws.t_mid, &ws.temb, &ws.temb_silu, &ws.out_scale}) grow(*p, H * 2 * 2);
            grow(ws.mod, 4 * H * 2 * 2);
        }
        ws.capacity = S;
    }
    dit_denoise(impl_->model->weights(), text_emb, grid_h, grid_w, latent, ws, stream);
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
    impl->handles = detail::plan_dit_weights(binder);
    impl->plan    = binder.finish();
    return LoadPlan(std::move(impl));
}

std::unique_ptr<detail::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    auto impl = std::make_unique<detail::LoadedModel::Impl>();
    detail::bind_dit_weights(impl->weights, plan.impl_->handles, materialized);
    impl->backing = std::move(materialized);
    return std::unique_ptr<detail::LoadedModel>(new detail::LoadedModel(std::move(impl)));
}

std::unique_ptr<detail::ImageProgram> Package::create_program(const LoadedModel& model, DeviceContext& device) {
    auto impl = std::make_unique<detail::ImageProgram::Impl>();
    impl->model  = &model;
    impl->device = &device;
    return std::unique_ptr<detail::ImageProgram>(new detail::ImageProgram(std::move(impl)));
}

} // namespace ninfer::targets::qwen_image_21
