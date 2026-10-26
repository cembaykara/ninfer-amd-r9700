#pragma once

// Block-causal multi-head attention over the joint [text (M), condition (Nc), target (Nt)] sequence.
// Text queries are causal; condition queries attend to text + condition; target queries attend to
// every key. cond_end = M + Nc (the end of the condition block; = M for text-to-image).
// q, k, v, out: {128, heads, tokens} BF16; out must not alias the inputs.
// Scores and values go through the gfx1201 BF16 WMMA, same as the vision encoder.
#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::targets::qwen_image_21::detail {

void dit_mha(const Tensor& q, const Tensor& k, const Tensor& v, int M, int cond_end, Tensor& out, hipStream_t stream);

} // namespace ninfer::targets::qwen_image_21::detail
