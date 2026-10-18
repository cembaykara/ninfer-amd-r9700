#pragma once

// One denoise step of the Qwen-Image-2.1 DiT: the 32-layer single-stream forward over the
// joint [text, image] sequence, producing the flow-matching velocity for the image tokens.
#include "config.h"
#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::targets::qwen_image_21::detail {

// The DiT weights, bound from the .ninfer artifact. Matrices are Weight (direct BF16 for now);
// the small per-head/text norms are plain tensors.
struct DitWeights {
    Weight img_in;                 // [4096, 64]
    Weight txt_in_in_layer;        // [4096, 4096]
    Weight txt_in_out_layer;       // [4096, 4096]
    Tensor txt_in_text_norm;       // [4096]
    Weight time_embed_linear_1;    // [4096, 256]
    Weight time_embed_linear_2;    // [4096, 4096]
    Weight modulation;             // [16384, 4096]
    Weight to_q[DitConfig::layers];
    Weight to_k[DitConfig::layers];
    Weight to_v[DitConfig::layers];
    Weight to_out[DitConfig::layers];
    Tensor norm_q[DitConfig::layers];  // [128]
    Tensor norm_k[DitConfig::layers];  // [128]
    Weight mlp_gate[DitConfig::layers];
    Weight mlp_up[DitConfig::layers];
    Weight mlp_down[DitConfig::layers];
    Weight norm_out_linear;        // [4096, 4096]
    Weight proj_out;               // [64, 4096]
};

// Pre-allocated activation workspace for one step (sized for the given M, N).
struct DitWorkspace {
    hip_bfloat16* img  = nullptr;  // [1, N, 4096]
    hip_bfloat16* txt  = nullptr;  // [1, M, 4096]
    hip_bfloat16* hs   = nullptr;  // [1, M+N, 4096]
    hip_bfloat16* hs2  = nullptr;  // [1, M+N, 4096]
    hip_bfloat16* temb = nullptr;  // [1, 4096]
    hip_bfloat16* temb_mid = nullptr;  // [1, 4096] timestep-embed intermediate
    hip_bfloat16* mod  = nullptr;  // [1, 16384]
    hip_bfloat16* q    = nullptr;  // [1, M+N, 4096]
    hip_bfloat16* k    = nullptr;  // [1, M+N, 4096]
    hip_bfloat16* v    = nullptr;  // [1, M+N, 4096]
    hip_bfloat16* attn = nullptr;  // [1, M+N, 4096]
    hip_bfloat16* x1   = nullptr;  // [1, M+N, 4096]
    hip_bfloat16* x2   = nullptr;  // [1, M+N, 4096]
    hip_bfloat16* mlp  = nullptr;  // [1, M+N, 4096]
    hip_bfloat16* out  = nullptr;  // [1, M+N, 64]
    float* freqs       = nullptr;  // [M+N, 128] RoPE freqs (re/im interleaved)
    hip_bfloat16* latent   = nullptr;  // [1, N, 64] current noisy latent
    hip_bfloat16* velocity = nullptr;  // [1, N, 64] model velocity
};

void dit_forward_step(DitWeights& w, const Tensor& latent, const Tensor& text_emb, float timestep,
                      Tensor& velocity, DitWorkspace& ws, int M, int N, hipStream_t stream);

// Run the full 8-step flow-matching denoise: init noise, 8 Euler steps, return the latent.
void dit_denoise(DitWeights& w, const Tensor& text_emb, int M, int N, Tensor& latent_out,
                     DitWorkspace& ws, hipStream_t stream);

} // namespace ninfer::targets::qwen_image_21::detail
