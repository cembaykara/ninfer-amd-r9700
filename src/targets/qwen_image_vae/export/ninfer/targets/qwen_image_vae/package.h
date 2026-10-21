#pragma once

// The Qwen-Image-2.1 VAE decoder target: latent [1,64,H/16,W/16] -> image [1,4,H,W].
// A diffusion target (like the DiT), not an autoregressive one. The entry point is
// a decode (latent -> image).
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

namespace targets::qwen_image_vae {

struct Package;

namespace detail {

struct VaeWeights;  // defined in impl/vae_forward.h

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
    friend struct qwen_image_vae::Package;
};

class LoadedModel {
public:
    ~LoadedModel();
    [[nodiscard]] const detail::VaeWeights& weights() const;
    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct qwen_image_vae::Package;
};

// The image-decode entry point: latent [1,64,H/16,W/16] -> image [1,4,H,W].
class ImageProgram {
public:
    ~ImageProgram();
    ImageProgram(const ImageProgram&)            = delete;
    ImageProgram& operator=(const ImageProgram&) = delete;

    // latent: [1, 64, H/16, W/16]; writes the decoded image [1, 4, H, W] to out_image.
    void decode(const Tensor& latent, Tensor& out_image, hipStream_t stream) const;

private:
    class Impl;
    explicit ImageProgram(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct qwen_image_vae::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen-image-21-vae-decoder";
    static constexpr std::string_view target_key = "qwen_image_vae_r9700";

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

} // namespace targets::qwen_image_vae
} // namespace ninfer
