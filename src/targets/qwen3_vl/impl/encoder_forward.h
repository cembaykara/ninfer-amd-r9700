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

// Pre-allocated activation workspace for one prefill (sized for the given seq).
struct EncoderWorkspace {
    hip_bfloat16* x    = nullptr;  // [1, seq, 4096] residual stream
    hip_bfloat16* xn   = nullptr;  // [1, seq, 4096] normalized
    hip_bfloat16* q    = nullptr;  // [1, seq, 32, 128]
    hip_bfloat16* k    = nullptr;  // [1, seq, 8, 128]
    hip_bfloat16* v    = nullptr;  // [1, seq, 8, 128]
    hip_bfloat16* attn = nullptr;  // [1, seq, 32, 128]
    hip_bfloat16* gate = nullptr;  // [1, seq, 12288]
    hip_bfloat16* up   = nullptr;  // [1, seq, 12288]
    hip_bfloat16* mlp  = nullptr;  // [1, seq, 12288]
    int* positions     = nullptr;  // [seq]
};

// Run the one-shot prefill: ids [1, seq] I32 -> hidden [1, seq, 4096] BF16.
void encoder_forward(const EncoderWeights& w, const Tensor& ids, Tensor& hidden,
                     EncoderWorkspace& ws, int seq, hipStream_t stream);

} // namespace ninfer::targets::qwen3_vl::detail
