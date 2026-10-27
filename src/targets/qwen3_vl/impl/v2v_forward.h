#pragma once

// v2v text-stack forward: embed the prompt, interleave the ViT visual tokens, run the 36-layer
// stack with the deepstack features injected at layers 0/1/2, return the combined embeddings.
#include "encoder_forward.h"
#include "core/tensor.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen3_vl::detail {

// ids {seq} I32 -> hidden {4096, seq} BF16 (token-major rows, pre-final-norm).
// visual_embeds [visual_count, 4096] (the ViT merged tokens); the contiguous block
// [visual_start, visual_start+visual_count) of the prompt is replaced by them.
// deepstack[3] are [visual_count, 4096] (the ViT deepstack features), injected at text layers 0/1/2.
void encoder_forward_v2v(const EncoderWeights& w, const Tensor& ids, Tensor& hidden,
                         EncoderWorkspace& ws, int seq,
                         const hip_bfloat16* visual_embeds, int visual_start, int visual_count,
                         const hip_bfloat16* const deepstack[3], hipStream_t stream);

// Multi-ref: visual_embeds is the concat of each image's V tokens; visual_starts/counts
// locate each pad block in `ids`. deepstack stays unused (pass nullptr) today.
void encoder_forward_v2v_multi(const EncoderWeights& w, const Tensor& ids, Tensor& hidden,
                               EncoderWorkspace& ws, int seq,
                               const hip_bfloat16* visual_embeds, const int* visual_starts,
                               const int* visual_counts, int n_visual,
                               const hip_bfloat16* const deepstack[3], hipStream_t stream);

} // namespace ninfer::targets::qwen3_vl::detail
