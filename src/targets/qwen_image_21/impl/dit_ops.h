#pragma once

// Target-local ops for the Qwen-Image-2.1 DiT that have no equivalent in ninfer/ops.
// Activations follow the ninfer convention: {hidden, tokens}, per-head {head_dim, heads, tokens}.
//
// The joint sequence is [text (M tokens), image]. With causal_condition, text tokens modulate
// from the t=0 row and image tokens from the real-timestep row; `img` / `txt` below are those rows.
#include "core/tensor.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen_image_21::detail {

// Interleaved-pair complex RoPE on q and k {128, heads, tokens} in place.
// freqs: [tokens][64] (cos, sin) FP32 pairs.
void dit_rope(const float* freqs, Tensor& q, Tensor& k, hipStream_t stream);

// No-affine LayerNorm over ne[0]. x and out must not alias.
void dit_layer_norm(const Tensor& x, float eps, Tensor& out, hipStream_t stream);

// x *= (1 + scale) in place; x {hidden, tokens}; rows < M use `txt`, the rest `img`.
void dit_modulate(Tensor& x, const hip_bfloat16* img, const hip_bfloat16* txt, int M,
                  hipStream_t stream);

// x += tanh(gate) * delta in place; gate row chosen like dit_modulate.
void dit_residual_gate(Tensor& x, const hip_bfloat16* img, const hip_bfloat16* txt, int M,
                       const Tensor& delta, hipStream_t stream);

// out = x * sigmoid(x). x and out may alias.
void dit_silu(const Tensor& x, Tensor& out, hipStream_t stream);

// latent += dt * velocity (FP32 math, BF16 storage).
void dit_euler(Tensor& latent, const Tensor& velocity, float dt, hipStream_t stream);

// out[n,t] = sum_k weight[n,k] * x[k,t]. Same WMMA fragment contract as the shared
// BF16 prefill kernel. A workgroup owns 128 rows by 512 tokens and stages one K
// slice of that weight tile, so each weight is read once per 512 tokens.
void dit_linear(const Tensor& x, const Weight& weight, Tensor& out, hipStream_t stream);

} // namespace ninfer::targets::qwen_image_21::detail
