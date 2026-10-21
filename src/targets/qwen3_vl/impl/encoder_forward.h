#pragma once

// One-shot prefill of the Qwen3-VL text encoder: embed the prompt, run the 36-layer
// stack, return the final hidden states [1, seq, 4096] (the text embeddings for the DiT).
#include "config.h"
#include "core/tensor.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen3_vl::detail {

// The text-encoder weights, bound from the .ninfer artifact.
struct EncoderWeights {
    Weight embed_tokens;                       // [vocab, 4096]
    Weight q_proj[EncoderConfig::layers];      // [4096, 4096]
    Weight k_proj[EncoderConfig::layers];      // [1024, 4096]
    Weight v_proj[EncoderConfig::layers];      // [1024, 4096]
    Weight o_proj[EncoderConfig::layers];      // [4096, 4096]
    Tensor q_norm[EncoderConfig::layers];      // [128]
    Tensor k_norm[EncoderConfig::layers];      // [128]
    Weight gate_proj[EncoderConfig::layers];   // [12288, 4096]
    Weight up_proj[EncoderConfig::layers];     // [12288, 4096]
    Weight down_proj[EncoderConfig::layers];   // [4096, 12288]
    Tensor input_layernorm[EncoderConfig::layers];      // [4096]
    Tensor post_attention_layernorm[EncoderConfig::layers]; // [4096]
    Tensor final_norm;                         // [4096]
};

// Pre-allocated activation workspace for one prefill (sized for `capacity` tokens).
struct EncoderWorkspace {
    hip_bfloat16* x    = nullptr;  // {4096, seq} residual stream
    hip_bfloat16* xn   = nullptr;  // {4096, seq} normalized
    hip_bfloat16* q    = nullptr;  // {32*128, seq}
    hip_bfloat16* k    = nullptr;  // {8*128, seq}
    hip_bfloat16* v    = nullptr;  // {8*128, seq}
    hip_bfloat16* qn   = nullptr;  // {128, 32, seq} QK-normed + rotated q
    hip_bfloat16* kn   = nullptr;  // {128, 8, seq} QK-normed + rotated k
    hip_bfloat16* attn = nullptr;  // {128, 32, seq}
    hip_bfloat16* gate = nullptr;  // {12288, seq}
    hip_bfloat16* up   = nullptr;  // {12288, seq}
    hip_bfloat16* mlp  = nullptr;  // {12288, seq}
    int* positions     = nullptr;  // {seq}
    int capacity       = 0;
};

// Run the one-shot prefill: ids {seq} I32 -> hidden {4096, seq} BF16 (pre-final-norm).
void encoder_forward(const EncoderWeights& w, const Tensor& ids, Tensor& hidden,
                     EncoderWorkspace& ws, int seq, hipStream_t stream);

} // namespace ninfer::targets::qwen3_vl::detail
