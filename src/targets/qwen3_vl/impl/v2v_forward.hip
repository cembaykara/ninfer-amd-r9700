// v2v text-stack forward (wave32 gfx1201).
#include "v2v_forward.h"
#include "causal_gqa.h"
#include "v2v_kernels.h"
#include "core/device.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
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
    Tensor xn(ws.xn, DType::BF16, {H, seq});
    embedding(ids, w.embed_tokens, xn, stream);

    // 2. Interleave the visual tokens into x (the residual stream).
    Tensor x(ws.x, DType::BF16, {H, seq});
    visual_merge(static_cast<const hip_bfloat16*>(xn.data), visual_embeds, visual_start,
                 visual_count, seq, H, static_cast<hip_bfloat16*>(x.data), stream);

    // 3. The 36-layer stack (with deepstack injection at layers 0/1/2).
    for (int l = 0; l < EncoderConfig::layers; l++) {
        rmsnorm(x, w.input_layernorm[l], eps, false, xn, stream);

        Tensor q(ws.q, DType::BF16, {heads * head_dim, seq});
        Tensor k(ws.k, DType::BF16, {kv_heads * head_dim, seq});
        Tensor v(ws.v, DType::BF16, {kv_heads * head_dim, seq});
        linear(xn, w.q_proj[l], q, stream);
        linear(xn, w.k_proj[l], k, stream);
        linear(xn, w.v_proj[l], v, stream);

        Tensor qn(ws.qn, DType::BF16, {head_dim, heads, seq});
        Tensor kn(ws.kn, DType::BF16, {head_dim, kv_heads, seq});
        rmsnorm(q.view({head_dim, heads, seq}), w.q_norm[l], eps, false, qn, stream);
        rmsnorm(k.view({head_dim, kv_heads, seq}), w.k_norm[l], eps, false, kn, stream);

        rope_mrope(qn, ws.positions, seq, (float)EncoderConfig::rope_theta, stream);
        rope_mrope(kn, ws.positions, seq, (float)EncoderConfig::rope_theta, stream);

        Tensor attn(ws.attn, DType::BF16, {head_dim, heads, seq});
        causal_gqa(qn, kn, v.view({head_dim, kv_heads, seq}), attn, stream);

        Tensor attn_flat = attn.view({heads * head_dim, seq});
        linear(attn_flat, w.o_proj[l], xn, stream);
        residual_add(xn, x, stream);

        rmsnorm(x, w.post_attention_layernorm[l], eps, false, xn, stream);

        Tensor gate(ws.gate, DType::BF16, {inter, seq});
        Tensor up(ws.up, DType::BF16, {inter, seq});
        Tensor mlp(ws.mlp, DType::BF16, {inter, seq});
        linear(xn, w.gate_proj[l], gate, stream);
        linear(xn, w.up_proj[l], up, stream);
        silu_mul(gate, up, mlp, stream);
        linear(mlp, w.down_proj[l], xn, stream);
        residual_add(xn, x, stream);

        // deepstack injection at text layers 0/1/2
        if (l < 3 && deepstack != nullptr) {
            deepstack_inject(reinterpret_cast<hip_bfloat16*>(x.data), deepstack[l],
                             visual_start, visual_count, seq, H, stream);
        }
    }

    // 4. The DiT consumes the last layer's output before the final RMSNorm (diffusers hooks
    // the norm out for both T2I and image-conditioned prompts), so w.final_norm is bound
    // but not applied, as in encoder_forward.
    HIP_CHECK(hipMemcpyAsync(hidden.data, x.data, (std::size_t)seq * H * sizeof(hip_bfloat16),
                             hipMemcpyDeviceToDevice, stream));
}

} // namespace ninfer::targets::qwen3_vl::detail
