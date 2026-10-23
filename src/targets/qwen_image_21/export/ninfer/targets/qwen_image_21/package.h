#pragma once

// The Qwen-Image-2.1-Turbo DiT target: a diffusion (text-to-image) target, not an
// autoregressive one. No KV cache, no sequence planning, no sampling program. The
// entry point is an image denoise (prompt embeddings + shape -> latent).
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

namespace targets::qwen_image_21 {

struct Package;

namespace detail {

struct DitWeights;  // defined in impl/dit_forward.h

enum class WeightsProfile : std::uint8_t {
    R9700Bf16,  // Piece 2a: direct BF16 (FP8/Q4 layered on later)
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
    friend struct qwen_image_21::Package;
};

class LoadedModel {
public:
    ~LoadedModel();
    [[nodiscard]] const detail::DitWeights& weights() const;
    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(const LoadedModel&&) = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct qwen_image_21::Package;
};

// The image-generation entry point: runs the 8-step denoise for one image.
class ImageProgram {
public:
    ~ImageProgram();
    ImageProgram(const ImageProgram&)            = delete;
    ImageProgram& operator=(const ImageProgram&) = delete;

    // text_emb: {4096, M} (from the text encoder); the image is grid_h x grid_w latent tokens.
    // latent {64, grid_h*grid_w} holds the initial noise and is denoised in place.
    void denoise(const Tensor& text_emb, int grid_h, int grid_w, Tensor& latent,
                 hipStream_t stream, float strength = 1.0f) const;

private:
    class Impl;
    explicit ImageProgram(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct qwen_image_21::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen-image-2.1-turbo";
    static constexpr std::string_view target_key = "qwen_image_21_r9700";

    using WeightsProfile = detail::WeightsProfile;
    using LoadPlan       = detail::LoadPlan;
    using LoadedModel    = detail::LoadedModel;
    using ImageProgram   = detail::ImageProgram;

    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static std::unique_ptr<ImageProgram>
    create_program(const LoadedModel& model, DeviceContext& device);
};

} // namespace targets::qwen_image_21
} // namespace ninfer
