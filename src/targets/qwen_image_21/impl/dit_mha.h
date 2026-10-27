#pragma once

// Block-causal multi-head attention over the joint sequence.
//
// HF QwenImage21: (q_idx >= kv_idx) | same_image_block. Text has no image id (causal
// only). Each condition image and the target are separate bidirectional blocks.
//
// `img_block_starts`/`img_block_ends` length `n_img` are half-open ranges of the
// condition-image blocks in joint coordinates. Tokens at >= `target_start` are the
// target block. Everything else is text.
//
// q, k, v, out: {128, heads, tokens} BF16; out must not alias the inputs.
#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::targets::qwen_image_21::detail {

void dit_mha(const Tensor& q, const Tensor& k, const Tensor& v, const int* img_block_starts,
             const int* img_block_ends, int n_img, int target_start, Tensor& out,
             hipStream_t stream);

} // namespace ninfer::targets::qwen_image_21::detail
