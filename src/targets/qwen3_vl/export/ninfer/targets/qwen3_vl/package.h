#pragma once

// The Qwen3-VL text encoder target: a one-shot prefill (prompt ids -> text embeddings).
// Not autoregressive: no KV cache, no sequence planning, no sampling program.
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "core/tensor.h"
#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::qwen3_vl {

struct Package;

namespace detail {

struct EncoderWeights;  // defined in impl/encoder_forward.h

enum class WeightsProfile : std::uint8_t {
    R9700Bf16,
};

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct qwen3_vl::Package;
};

class LoadedModel {
public:
    ~LoadedModel();
    [[nodiscard]] const detail::EncoderWeights& weights() const;
    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(const LoadedModel&&) = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct qwen3_vl::Package;
};

// The text-encoder entry point: runs the one-shot prefill for one prompt.
class EncoderProgram {
public:
    ~EncoderProgram();
    EncoderProgram(const EncoderProgram&)            = delete;
    EncoderProgram& operator=(const EncoderProgram&) = delete;

    // ids: [1, seq] I32. Writes the final hidden states [1, seq, 4096] to out_hidden.
    void embed(const Tensor& ids, Tensor& out_hidden, hipStream_t stream) const;

private:
    class Impl;
    explicit EncoderProgram(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct qwen3_vl::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen3-vl-text-encoder";
    static constexpr std::string_view target_key = "qwen3_vl_r9700";

    using WeightsProfile = detail::WeightsProfile;
    using LoadPlan       = detail::LoadPlan;
    using LoadedModel    = detail::LoadedModel;
    using EncoderProgram = detail::EncoderProgram;

    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static std::unique_ptr<EncoderProgram>
    create_program(const LoadedModel& model, DeviceContext& device);
};

} // namespace targets::qwen3_vl
} // namespace ninfer
