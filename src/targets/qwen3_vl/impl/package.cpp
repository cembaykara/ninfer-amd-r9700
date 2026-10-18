#include "ninfer/targets/qwen3_vl/package.h"

#include "targets/qwen3_vl/impl/encoder_forward.h"
#include "targets/qwen3_vl/impl/load/bindings.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/device.h"

#include <hip/hip_runtime.h>

#include <memory>

namespace ninfer::targets::qwen3_vl {
namespace detail {

struct LoadPlan::Impl {
    artifact::MaterializationPlan plan;
    EncoderWeightHandles handles;
};

struct LoadedModel::Impl {
    EncoderWeights weights;
};

struct EncoderProgram::Impl {
    const LoadedModel* model = nullptr;
    DeviceContext* device    = nullptr;
    EncoderWorkspace workspace;
    bool workspace_ready = false;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan() = default;
const artifact::MaterializationPlan& LoadPlan::materialization() const { return impl_->plan; }

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;
const EncoderWeights& LoadedModel::weights() const { return impl_->weights; }

EncoderProgram::EncoderProgram(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
EncoderProgram::~EncoderProgram() = default;

void EncoderProgram::embed(const Tensor& ids, Tensor& out_hidden, hipStream_t stream) const {
    const int seq = ids.ne[1];
    if (!impl_->workspace_ready) {
        auto& ws = impl_->workspace;
        const int H = EncoderConfig::hidden;
        const int heads = EncoderConfig::heads;
        const int kv_heads = EncoderConfig::kv_heads;
        const int head_dim = EncoderConfig::head_dim;
        const int inter = EncoderConfig::intermediate;
        auto alloc = [&](std::size_t bytes) {
            void* p = nullptr;
            HIP_CHECK(hipMalloc(&p, bytes));
            return p;
        };
        ws.x    = (hip_bfloat16*)alloc((std::size_t)seq * H * 2);
        ws.xn   = (hip_bfloat16*)alloc((std::size_t)seq * H * 2);
        ws.q    = (hip_bfloat16*)alloc((std::size_t)seq * heads * head_dim * 2);
        ws.k    = (hip_bfloat16*)alloc((std::size_t)seq * kv_heads * head_dim * 2);
        ws.v    = (hip_bfloat16*)alloc((std::size_t)seq * kv_heads * head_dim * 2);
        ws.attn = (hip_bfloat16*)alloc((std::size_t)seq * heads * head_dim * 2);
        ws.gate = (hip_bfloat16*)alloc((std::size_t)seq * inter * 2);
        ws.up   = (hip_bfloat16*)alloc((std::size_t)seq * inter * 2);
        ws.mlp  = (hip_bfloat16*)alloc((std::size_t)seq * inter * 2);
        ws.positions = (int*)alloc((std::size_t)seq * 4);
        // fill positions [0, 1, 2, ...] on device
        int* hpos = new int[seq];
        for (int i = 0; i < seq; i++) hpos[i] = i;
        HIP_CHECK(hipMemcpy(ws.positions, hpos, (std::size_t)seq * 4, hipMemcpyHostToDevice));
        delete[] hpos;
        impl_->workspace_ready = true;
    }
    encoder_forward(impl_->model->weights(), ids, out_hidden, impl_->workspace, seq, stream);
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
    impl->plan    = binder.finish();
    return LoadPlan(std::move(impl));
}

std::unique_ptr<detail::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    auto impl = std::make_unique<detail::LoadedModel::Impl>();
    detail::bind_encoder_weights(impl->weights, plan.impl_->handles, materialized);
    return std::unique_ptr<detail::LoadedModel>(new detail::LoadedModel(std::move(impl)));
}

std::unique_ptr<detail::EncoderProgram> Package::create_program(const LoadedModel& model, DeviceContext& device) {
    auto impl = std::make_unique<detail::EncoderProgram::Impl>();
    impl->model  = &model;
    impl->device = &device;
    return std::unique_ptr<detail::EncoderProgram>(new detail::EncoderProgram(std::move(impl)));
}

} // namespace ninfer::targets::qwen3_vl
