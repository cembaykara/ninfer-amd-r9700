#pragma once

// The Qwen-Image-2.1 DiT: the 32-layer single-stream forward over the joint [text, image]
// sequence and the flow-matching denoise loop around it.
//
// Tensors use the ninfer convention: ne[0] is the innermost extent, so S tokens of H features
// are {H, S} and per-head tensors are {head_dim, heads, S}.
#include "config.h"
#include "core/tensor.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen_image_21::detail {

// The DiT weights, bound from the .ninfer artifact.
struct DitWeights {
    Weight img_in;                 // [4096, 64]
    Weight txt_in_in_layer;        // [4096, 4096]
    Weight txt_in_out_layer;       // [4096, 4096]
    Tensor txt_in_text_norm;       // [4096], zero-centred
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

// Activation workspace. Sequence buffers hold `capacity` tokens; the conditioning buffers have
// two columns: 0 = the real timestep (image tokens), 1 = t=0 (text tokens).
struct DitWorkspace {
    int capacity = 0;
    hip_bfloat16* hs   = nullptr;  // {H, S}
    hip_bfloat16* x1   = nullptr;  // {H, S}
    hip_bfloat16* x2   = nullptr;  // {H, S}
    hip_bfloat16* q    = nullptr;  // {H, S}
    hip_bfloat16* k    = nullptr;  // {H, S}
    hip_bfloat16* v    = nullptr;  // {H, S}
    hip_bfloat16* qn   = nullptr;  // {H, S}
    hip_bfloat16* kn   = nullptr;  // {H, S}
    hip_bfloat16* attn = nullptr;  // {H, S}
    hip_bfloat16* gate = nullptr;  // {I, S}
    hip_bfloat16* up   = nullptr;  // {I, S}
    hip_bfloat16* act  = nullptr;  // {I, S}
    hip_bfloat16* txt  = nullptr;  // {H, S} projected text, computed once per request
    float* freqs       = nullptr;  // [S][64] (cos, sin)
    hip_bfloat16* velocity = nullptr;  // {64, S}
    hip_bfloat16* t_sin     = nullptr;  // {256, 2}
    hip_bfloat16* t_mid     = nullptr;  // {H, 2}
    hip_bfloat16* temb      = nullptr;  // {H, 2}
    hip_bfloat16* temb_silu = nullptr;  // {H, 2}
    hip_bfloat16* mod       = nullptr;  // {4H, 2}: scale1, gate1, scale2, gate2
    hip_bfloat16* out_scale = nullptr;  // {H, 2}
};

// Denoise the target latent {64, Nt} in place. text_emb is {4096, M}; the target is grid_h x grid_w
// tokens (Nt = grid_h * grid_w). cond_latent is the condition {64, Nc} (Nc = cond_h * cond_w);
// pass a zero-sized tensor for text-to-image. `latent` holds the initial noise on entry.
void dit_denoise(const DitWeights& w, const Tensor& text_emb, int cond_h, int cond_w, int grid_h, int grid_w,
                 const Tensor& cond_latent, Tensor& latent, DitWorkspace& ws, hipStream_t stream);

} // namespace ninfer::targets::qwen_image_21::detail
