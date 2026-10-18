#pragma once

#include "ninfer/types.h"
#include "runtime/engine/request_memory.h"
#include <ninfer/targets/qwen3_8_27b/package.h>
#include <ninfer/targets/qwen_image_21/package.h>
#include <ninfer/targets/qwen3_vl/package.h>
#include <ninfer/targets/qwen_image_vae/package.h>

#include <memory>
#include <optional>
#include <string_view>
#include <variant>

namespace ninfer {

struct DeviceContext;

namespace targets {

[[nodiscard]] constexpr std::optional<std::string_view>
registered_target_key(std::string_view model_id) noexcept {
    if (model_id == qwen3_8_27b::Package::model_id) { return qwen3_8_27b::Package::target_key; }
    if (model_id == qwen_image_21::Package::model_id) { return qwen_image_21::Package::target_key; }
    if (model_id == qwen3_vl::Package::model_id) { return qwen3_vl::Package::target_key; }
    if (model_id == qwen_image_vae::Package::model_id) { return qwen_image_vae::Package::target_key; }
    return std::nullopt;
}

struct LoadedQwen3_8_27B {
    std::unique_ptr<qwen3_8_27b::Package::LoadedModel> model;
    qwen3_8_27b::Package::Frontend frontend;

    explicit LoadedQwen3_8_27B(std::unique_ptr<qwen3_8_27b::Package::LoadedModel> stable_model);
    ~LoadedQwen3_8_27B();

    LoadedQwen3_8_27B(const LoadedQwen3_8_27B&)            = delete;
    LoadedQwen3_8_27B& operator=(const LoadedQwen3_8_27B&) = delete;
};

struct Qwen3_8_27BInstance {
    using Package = qwen3_8_27b::Package;

    std::unique_ptr<LoadedQwen3_8_27B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Package::Program> program;

    Qwen3_8_27BInstance(std::unique_ptr<LoadedQwen3_8_27B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Package::SequencePlan sequence_plan, DeviceContext& device,
                        std::unique_ptr<HostPinnedArena> kv_ram_arena);
    ~Qwen3_8_27BInstance();

    Qwen3_8_27BInstance(const Qwen3_8_27BInstance&)            = delete;
    Qwen3_8_27BInstance& operator=(const Qwen3_8_27BInstance&) = delete;
    Qwen3_8_27BInstance(Qwen3_8_27BInstance&&) = default;  // needed for the variant ActiveTarget
};

struct LoadedDit {
    std::unique_ptr<qwen_image_21::Package::LoadedModel> model;
    std::unique_ptr<qwen_image_21::Package::ImageProgram> program;
    explicit LoadedDit(std::unique_ptr<qwen_image_21::Package::LoadedModel> m, DeviceContext& device);
    ~LoadedDit();
    LoadedDit(const LoadedDit&)            = delete;
    LoadedDit& operator=(const LoadedDit&) = delete;
};

struct DitInstance {
    std::unique_ptr<LoadedDit> loaded;
    DitInstance(std::unique_ptr<LoadedDit> l);
    ~DitInstance();
    DitInstance(const DitInstance&)            = delete;
    DitInstance& operator=(const DitInstance&) = delete;
    DitInstance(DitInstance&&) = default;  // needed for the variant ActiveTarget
};

struct LoadedEncoder {
    std::unique_ptr<qwen3_vl::Package::LoadedModel> model;
    std::unique_ptr<qwen3_vl::Package::EncoderProgram> program;
    explicit LoadedEncoder(std::unique_ptr<qwen3_vl::Package::LoadedModel> m, DeviceContext& device);
    ~LoadedEncoder();
    LoadedEncoder(const LoadedEncoder&)            = delete;
    LoadedEncoder& operator=(const LoadedEncoder&) = delete;
};

struct EncoderInstance {
    std::unique_ptr<LoadedEncoder> loaded;
    EncoderInstance(std::unique_ptr<LoadedEncoder> l);
    ~EncoderInstance();
    EncoderInstance(const EncoderInstance&)            = delete;
    EncoderInstance& operator=(const EncoderInstance&) = delete;
    EncoderInstance(EncoderInstance&&) = default;  // needed for the variant ActiveTarget
};

struct LoadedVae {
    std::unique_ptr<qwen_image_vae::Package::LoadedModel> model;
    std::unique_ptr<qwen_image_vae::Package::ImageProgram> program;
    explicit LoadedVae(std::unique_ptr<qwen_image_vae::Package::LoadedModel> m, DeviceContext& device);
    ~LoadedVae();
    LoadedVae(const LoadedVae&)            = delete;
    LoadedVae& operator=(const LoadedVae&) = delete;
};

struct VaeInstance {
    std::unique_ptr<LoadedVae> loaded;
    VaeInstance(std::unique_ptr<LoadedVae> l);
    ~VaeInstance();
    VaeInstance(const VaeInstance&)            = delete;
    VaeInstance& operator=(const VaeInstance&) = delete;
    VaeInstance(VaeInstance&&) = default;  // needed for the variant ActiveTarget
};

using ActiveTarget = std::unique_ptr<std::variant<Qwen3_8_27BInstance, DitInstance, EncoderInstance, VaeInstance>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

} // namespace targets
} // namespace ninfer
