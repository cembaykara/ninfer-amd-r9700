// One-shot prefill of the Qwen3-VL text encoder (wave32 gfx1201).
#include "encoder_forward.h"
#include "causal_gqa.h"
#include "core/device.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/silu_mul.h"

#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen3_vl::detail {
using namespace ninfer::ops;

void encoder_forward(const EncoderWeights& w, const Tensor& ids, Tensor& hidden,
                     EncoderWorkspace& ws, int seq, hipStream_t stream) {
    const int H        = EncoderConfig::hidden;
    const int heads    = EncoderConfig::heads;
    const int kv_heads = EncoderConfig::kv_heads;
    const int head_dim = EncoderConfig::head_dim;
    const int inter    = EncoderConfig::intermediate;
    const float eps    = EncoderConfig::eps;

    // 1. Embed the prompt: ids [1, seq] -> x [1, seq, H].
    Tensor x(ws.x, DType::BF16, {1, seq, H});
    embedding(ids, w.embed_tokens, x, stream);

    // 2. The 36-layer stack.
    for (int l = 0; l < EncoderConfig::layers; l++) {
        // RMSNorm (input_layernorm).
        Tensor xn(ws.xn, DType::BF16, {1, seq, H});
        rmsnorm(x, w.input_layernorm[l], eps, false, xn, stream);

        // QKV projection.
        Tensor q(ws.q, DType::BF16, {1, seq, heads, head_dim});
        Tensor k(ws.k, DType::BF16, {1, seq, kv_heads, head_dim});
        Tensor v(ws.v, DType::BF16, {1, seq, kv_heads, head_dim});
        linear(xn, w.q_proj[l], q, stream);
        linear(xn, w.k_proj[l], k, stream);
        linear(xn, w.v_proj[l], v, stream);

        // QK-norm (in place).
        rmsnorm(q, w.q_norm[l], eps, false, q, stream);
        rmsnorm(k, w.k_norm[l], eps, false, k, stream);

        // RoPE (1-D; mRoPE collapses for text-only input).
        rope(Tensor(ws.positions, DType::I32, {seq}), head_dim,
             (float)EncoderConfig::rope_theta, q, k, stream);

        // Causal GQA attention.
        Tensor attn(ws.attn, DType::BF16, {1, seq, heads, head_dim});
        causal_gqa(q, k, v, attn, stream);

        // O projection + residual.
        linear(attn, w.o_proj[l], xn, stream);
        residual_add(xn, x, stream);

        // RMSNorm (post_attention_layernorm).
        rmsnorm(x, w.post_attention_layernorm[l], eps, false, xn, stream);

        // MLP: gate, up, silu_mul, down.
        Tensor gate(ws.gate, DType::BF16, {1, seq, inter});
        Tensor up(ws.up, DType::BF16, {1, seq, inter});
        Tensor mlp(ws.mlp, DType::BF16, {1, seq, inter});
        linear(xn, w.gate_proj[l], gate, stream);
        linear(xn, w.up_proj[l], up, stream);
        silu_mul(gate, up, mlp, stream);
        linear(mlp, w.down_proj[l], xn, stream);
        residual_add(xn, x, stream);
    }

    // 3. Final RMSNorm -> hidden [1, seq, H].
    rmsnorm(x, w.final_norm, eps, false, hidden, stream);
}

} // namespace ninfer::targets::qwen3_vl::detail
