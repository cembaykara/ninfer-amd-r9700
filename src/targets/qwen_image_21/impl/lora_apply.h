#pragma once

// One denoise with a LoRA file applied on the already-loaded DiT weights.
#include "dit_forward.h"

#include <filesystem>
#include <functional>

namespace ninfer::targets::qwen_image_21::detail {

// Adds scale * (B @ A) onto the loaded weights, runs `denoise`, then copies those tensors
// back from `artifact`. The .ninfer file is only read. An exception from `denoise` still
// restores the weights. Throws std::invalid_argument for a bad LoRA file or a non-finite scale.
void denoise_with_lora(const DitWeights& weights, const std::filesystem::path& artifact,
                       const std::filesystem::path& lora, float scale, hipStream_t stream,
                       const std::function<void()>& denoise);

} // namespace ninfer::targets::qwen_image_21::detail
