#pragma once

// v2v glue kernels: interleave the ViT visual tokens into the prompt embeddings and
// inject the deepstack features at the visual positions (text layers 0/1/2).
#include "core/tensor.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen3_vl::detail {

// Replace the contiguous visual block [visual_start, visual_start+visual_count) of the text
// embeddings with the ViT visual tokens. Other positions copy through unchanged.
// text_embeds [seq, H], visual_embeds [visual_count, H], out [seq, H].
void visual_merge(const hip_bfloat16* text_embeds, const hip_bfloat16* visual_embeds,
                  int visual_start, int visual_count, int seq, int hidden,
                  hip_bfloat16* out, hipStream_t stream);

// Add the deepstack feature to the contiguous visual block of the hidden states (in place).
// hidden [seq, H], feature [visual_count, H].
void deepstack_inject(hip_bfloat16* hidden, const hip_bfloat16* feature,
                      int visual_start, int visual_count, int seq, int hidden_dim,
                      hipStream_t stream);

} // namespace ninfer::targets::qwen3_vl::detail
