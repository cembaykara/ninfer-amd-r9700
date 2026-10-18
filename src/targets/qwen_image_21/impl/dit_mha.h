#pragma once

// Full bidirectional multi-head attention over the joint [text, image] sequence.
// q, k, v, out: [batch, tokens, heads, head_dim] BF16. Each query attends to every key.
// Simple (non-flash) two-pass kernel: correct for validation; flash-attention for production.
#include "core/tensor.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen_image_21::detail {

void dit_mha(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out, hipStream_t stream);

} // namespace ninfer::targets::qwen_image_21::detail
