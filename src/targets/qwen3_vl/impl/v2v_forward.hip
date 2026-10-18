// v2v text-stack forward (wave32 gfx1201).
#include "v2v_forward.h"
#include "causal_gqa.h"
#include "v2v_kernels.h"
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

void encoder_forward_v2v(const EncoderWeights& w, const Tensor& ids, Tensor& hidden,
                         EncoderWorkspace& ws, int seq,
                         const hip_bfloat16* visual_embeds, int visual_start, int visual_count,
                         const hip_bfloat16* const deepstack[3], hipStream_t stream) {
    const int H        = EncoderConfig::hidden;
    const int heads    = EncoderConfig::heads;
    const int kv_heads = EncoderConfig::kv_heads;
    const int head_dim = EncoderConfig::head_dim;
    const int inter    = EncoderConfig::intermediate;
    const float eps    = EncoderConfig::eps;

    // 1. Embed the prompt into xn (text embeddings).
    Tensor xn(ws.xn, DType::BF16, {1, seq, H});
    embedding(ids, w.embed_tokens, xn, stream);

    // 2. Interleave the visual tokens into x (the residual stream).
    Tensor x(ws.x, DType::BF16, {1, seq, H});
    visual_merge(reinterpret_cast<const hip_bfloat16*>(xn.data), visual_embeds,
                 visual_start, visual_count, seq, H,
                 reinterpret_cast<hip_bfloat16*>(x.data), stream);

    // 3. The 36-layer stack (with deepstack injection at layers 0/1/2).
    for (int l = 0; l < EncoderConfig::layers; l++) {
        Tensor xn_t(ws.xn, DType::BF16, {1, seq, H});
        rmsnorm(x, w.input_layernorm[l], eps, false, xn_t, stream);

        Tensor q(ws.q, DType::BF16, {1, seq, heads, head_dim});
        Tensor k(ws.k, DType::BF16, {1, seq, kv_heads, head_dim});
        Tensor v(ws.v, DType::BF16, {1, seq, kv_heads, head_dim});
        linear(xn_t, w.q_proj[l], q, stream);
        linear(xn_t, w.k_proj[l], k, stream);
        linear(xn_t, w.v_proj[l], v, stream);

        rmsnorm(q, w.q_norm[l], eps, false, q, stream);
        rmsnorm(k, w.k_norm[l], eps, false, k, stream);

        rope(Tensor(ws.positions, DType::I32, {seq}), head_dim,
             (float)EncoderConfig::rope_theta, q, k, stream);

        Tensor attn(ws.attn, DType::BF16, {1, seq, heads, head_dim});
        causal_gqa(q, k, v, attn, stream);

        linear(attn, w.o_proj[l], xn_t, stream);
        residual_add(xn_t, x, stream);

        rmsnorm(x, w.post_attention_layernorm[l], eps, false, xn_t, stream);

        Tensor gate(ws.gate, DType::BF16, {1, seq, inter});
        Tensor up(ws.up, DType::BF16, {1, seq, inter});
        Tensor mlp(ws.mlp, DType::BF16, {1, seq, inter});
        linear(xn_t, w.gate_proj[l], gate, stream);
        linear(xn_t, w.up_proj[l], up, stream);
        silu_mul(gate, up, mlp, stream);
        linear(mlp, w.down_proj[l], xn_t, stream);
        residual_add(xn_t, x, stream);

        // deepstack injection at text layers 0/1/2
        if (l < 3 && deepstack != nullptr) {
            deepstack_inject(reinterpret_cast<hip_bfloat16*>(x.data), deepstack[l],
                             visual_start, visual_count, seq, H, stream);
        }
    }

    // 4. Final RMSNorm -> hidden [1, seq, H].
    rmsnorm(x, w.final_norm, eps, false, hidden, stream);
}

} // namespace ninfer::targets::qwen3_vl::detail
