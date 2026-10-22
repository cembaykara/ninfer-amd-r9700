// One-shot prefill of the Qwen3-VL text encoder (wave32 gfx1201).
//
// Tensors use the ninfer convention: ne[0] is the innermost (fastest) extent, so an activation
// of `seq` tokens with H features is {H, seq}, and per-head q/k/v are {head_dim, heads, seq}.
#include "encoder_forward.h"
#include "causal_gqa.h"
#include "core/device.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"

#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen3_vl::detail {
using namespace ninfer::ops;

namespace {

// Rotate-half RoPE over the full head_dim, in place. Text-only Qwen3-VL input has equal
// t/h/w positions, so interleaved mRoPE reduces to this 1-D form. The shared ops::rope only
// accepts the Qwen3.8 geometry (theta 1e7), hence a local kernel.
__global__ void rope_neox_kernel(hip_bfloat16* __restrict__ x, const int* __restrict__ positions,
                                 int heads, int head_dim, float theta) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int half  = head_dim / 2;
    const int i     = threadIdx.x;
    if (i >= half) { return; }
    hip_bfloat16* row   = x + ((std::int64_t)token * heads + head) * head_dim;
    const float inv_freq = powf(theta, -2.0F * (float)i / (float)head_dim);
    const float angle    = (float)positions[token] * inv_freq;
    const float c        = cosf(angle);
    const float s        = sinf(angle);
    const float x1       = (float)row[i];
    const float x2       = (float)row[i + half];
    row[i]               = hip_bfloat16(x1 * c - x2 * s);
    row[i + half]        = hip_bfloat16(x2 * c + x1 * s);
}

} // namespace

void rope_neox(Tensor& x, const int* positions, float theta, hipStream_t stream) {
    const int head_dim = x.ne[0];
    const int heads    = x.ne[1];
    const int tokens   = x.ne[2];
    dim3 grid(tokens, heads);
    rope_neox_kernel<<<grid, dim3(head_dim / 2), 0, stream>>>(
        static_cast<hip_bfloat16*>(x.data), positions, heads, head_dim, theta);
    HIP_CHECK(hipGetLastError());
}

void encoder_forward(const EncoderWeights& w, const Tensor& ids, Tensor& hidden,
                     EncoderWorkspace& ws, int seq, hipStream_t stream) {
    const int H        = EncoderConfig::hidden;
    const int heads    = EncoderConfig::heads;
    const int kv_heads = EncoderConfig::kv_heads;
    const int head_dim = EncoderConfig::head_dim;
    const int inter    = EncoderConfig::intermediate;
    const float eps    = EncoderConfig::eps;

    // 1. Embed the prompt: ids {seq} -> x {H, seq}.
    Tensor x(ws.x, DType::BF16, {H, seq});
    embedding(ids, w.embed_tokens, x, stream);

    Tensor xn(ws.xn, DType::BF16, {H, seq});
    Tensor q(ws.q, DType::BF16, {heads * head_dim, seq});
    Tensor k(ws.k, DType::BF16, {kv_heads * head_dim, seq});
    Tensor v(ws.v, DType::BF16, {kv_heads * head_dim, seq});
    Tensor qn(ws.qn, DType::BF16, {head_dim, heads, seq});
    Tensor kn(ws.kn, DType::BF16, {head_dim, kv_heads, seq});
    Tensor vh = v.view({head_dim, kv_heads, seq});
    Tensor attn(ws.attn, DType::BF16, {head_dim, heads, seq});
    Tensor attn_flat = attn.view({heads * head_dim, seq});
    Tensor gate(ws.gate, DType::BF16, {inter, seq});
    Tensor up(ws.up, DType::BF16, {inter, seq});
    Tensor mlp(ws.mlp, DType::BF16, {inter, seq});

    // 2. The 36-layer stack.
    for (int l = 0; l < EncoderConfig::layers; l++) {
        rmsnorm(x, w.input_layernorm[l], eps, false, xn, stream);

        linear(xn, w.q_proj[l], q, stream);
        linear(xn, w.k_proj[l], k, stream);
        linear(xn, w.v_proj[l], v, stream);

        // Per-head QK-norm over head_dim.
        rmsnorm(q.view({head_dim, heads, seq}), w.q_norm[l], eps, false, qn, stream);
        rmsnorm(k.view({head_dim, kv_heads, seq}), w.k_norm[l], eps, false, kn, stream);

        rope_neox(qn, ws.positions, (float)EncoderConfig::rope_theta, stream);
        rope_neox(kn, ws.positions, (float)EncoderConfig::rope_theta, stream);

        causal_gqa(qn, kn, vh, attn, stream);

        linear(attn_flat, w.o_proj[l], xn, stream);
        residual_add(xn, x, stream);

        rmsnorm(x, w.post_attention_layernorm[l], eps, false, xn, stream);
        linear(xn, w.gate_proj[l], gate, stream);
        linear(xn, w.up_proj[l], up, stream);
        silu_mul(gate, up, mlp, stream);
        linear(mlp, w.down_proj[l], xn, stream);
        residual_add(xn, x, stream);
    }

    // 3. The DiT consumes the last layer's output before the final RMSNorm (diffusers hooks
    // the norm out), so w.final_norm is bound but not applied.
    HIP_CHECK(hipMemcpyAsync(hidden.data, x.data, (std::size_t)seq * H * sizeof(hip_bfloat16),
                             hipMemcpyDeviceToDevice, stream));
}

} // namespace ninfer::targets::qwen3_vl::detail
