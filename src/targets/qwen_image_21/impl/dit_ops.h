#pragma once

// Target-local ops for the Qwen-Image-2.1 DiT that have no equivalent in ninfer/ops:
// 3D (frame/height/width) complex RoPE, no-affine LayerNorm, and the AdaLN modulation.
#include "core/tensor.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen_image_21::detail {

// Applies the precomputed complex RoPE frequencies to q and k in place.
// freqs: [tokens, 128] FP32 (64 complex pairs, re/im interleaved per pair).
// q, k:  [batch, tokens, heads, 128] BF16. Batch is applied by the caller over b.
void dit_rope(const float* freqs, Tensor& q, Tensor& k, hipStream_t stream);

// No-affine LayerNorm (elementwise_affine=false): out = (x - mean) / sqrt(var + eps).
void dit_layer_norm(const Tensor& x, float eps, Tensor& out, hipStream_t stream);

// AdaLN modulate: out = x * (1 + scale), where scale is [batch, hidden] broadcast over tokens.
void dit_modulate_scale(const Tensor& x, const Tensor& scale, Tensor& out, hipStream_t stream);

// Residual gate: out = x + tanh(gate) * delta, gate [batch, hidden] broadcast, delta [b,t,h].
void dit_residual_gate(const Tensor& x, const Tensor& gate, const Tensor& delta, Tensor& out,
                       hipStream_t stream);

} // namespace ninfer::targets::qwen_image_21::detail
