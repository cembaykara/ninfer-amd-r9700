#pragma once

// One denoise with LoRA files applied on the already-loaded DiT weights.
#include "dit_forward.h"

#include <filesystem>
#include <functional>
#include <vector>

namespace ninfer::targets::qwen_image_21::detail {

// One LoRA for a denoise: a safetensors path and its scale.
struct LoraRef {
    std::filesystem::path path;
    float scale = 1.F;
};

// Adds scale * (B @ A) for every LoRA onto the loaded weights, runs `denoise` once, then copies
// the touched tensors back from `artifact`. The .ninfer file is only read. An exception from
// `denoise` still restores the weights. Throws std::invalid_argument for a bad LoRA file or a
// non-finite scale.
void denoise_with_lora(const DitWeights& weights, const std::filesystem::path& artifact,
                       const std::vector<LoraRef>& loras, hipStream_t stream,
                       const std::function<void()>& denoise);

} // namespace ninfer::targets::qwen_image_21::detail
