#pragma once

// Read a diffusers-style Qwen-Image LoRA safetensors file into per-tensor patches.
// The file is only read. Each patch is one loaded DiT matrix: weight + scale * (B @ A),
// with img_mlp.gate_up split across mlp/gate and mlp/up.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::targets::qwen_image_21::detail {

enum class LoraSlot : std::uint8_t { Query, Key, Value, Output, Gate, Up, Down };

struct LoraPatch {
    int layer = 0;
    LoraSlot slot = LoraSlot::Query;
    int rows      = 0;
    int cols      = 0;
    int rank      = 0;
    // Row-major BF16. A is (rank, cols). B is (rows, rank), already the gate or up half.
    std::vector<std::uint16_t> a;
    std::vector<std::uint16_t> b;
    std::string tensor_name;
};

// Throws std::invalid_argument when the file is missing, not the expected LoRA, or incomplete.
[[nodiscard]] std::vector<LoraPatch> read_lora_patches(const std::filesystem::path& path);

[[nodiscard]] const char* lora_slot_suffix(LoraSlot slot) noexcept;

} // namespace ninfer::targets::qwen_image_21::detail
