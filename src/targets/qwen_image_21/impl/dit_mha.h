#pragma once

// Block-causal multi-head attention over the joint [text (M), image] sequence.
// Text queries are causal; image queries attend to every key (text and image).
// q, k, v, out: {128, heads, tokens} BF16; out must not alias the inputs.
// Scores and values go through the gfx1201 BF16 WMMA, same as the vision encoder.
#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::targets::qwen_image_21::detail {

void dit_mha(const Tensor& q, const Tensor& k, const Tensor& v, int M, Tensor& out, hipStream_t stream);

} // namespace ninfer::targets::qwen_image_21::detail
