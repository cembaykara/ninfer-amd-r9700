// One denoise step of the Qwen-Image-2.1 DiT (wave32 gfx1201).
#include "dit_forward.h"

#include "dit_mha.h"
#include "dit_ops.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include <hip/hip_bfloat16.h>

#include <cmath>
#include <vector>

namespace ninfer::targets::qwen_image_21::detail {
using namespace ninfer::ops;

void dit_forward_step(DitWeights& w, const Tensor& latent, const Tensor& text_emb, float timestep,
                      Tensor& velocity, DitWorkspace& ws, int M, int N, hipStream_t stream) {
    const int S = M + N;
    const int H = DitConfig::hidden;
    const int C = DitConfig::in_channels;

    // 1. img = linear(latent, img_in)  [1, N, H]
    Tensor img(ws.img, DType::BF16, {1, N, H});
    linear(latent, w.img_in, img, stream);

    // 2-4. txt = out_layer(gelu(in_layer(rmsnorm(text_emb, text_norm, zero_centered))))
    Tensor txt(ws.txt, DType::BF16, {1, M, H});
    rmsnorm(text_emb, w.txt_in_text_norm, DitConfig::eps, true, txt, stream);
    linear(txt, w.txt_in_in_layer, txt, stream);
    gelu(txt, GeluMode::Tanh, stream);
    linear(txt, w.txt_in_out_layer, txt, stream);

    // 5. hs = [txt | img]  [1, S, H]
    Tensor hs(ws.hs, DType::BF16, {1, S, H});
    hipMemcpyAsync(ws.hs, ws.txt, (std::size_t)M * H * 2, hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(ws.hs + (std::size_t)M * H, ws.img, (std::size_t)N * H * 2, hipMemcpyDeviceToDevice,
                   stream);

    // 6. temb = time_embed(timestep)  [1, H]
    {
        hip_bfloat16 sin_emb[256];
        for (int i = 0; i < 128; i++) {
            const float freq = std::exp(-std::log(10000.0F) * i / 128.0F);
            const float arg  = timestep * 1000.0F * freq;
            sin_emb[i]       = hip_bfloat16(std::cos(arg));
            sin_emb[128 + i] = hip_bfloat16(std::sin(arg));
        }
        hipMemcpyAsync(ws.temb, sin_emb, 256 * sizeof(hip_bfloat16), hipMemcpyHostToDevice, stream);
        Tensor sin_in(ws.temb, DType::BF16, {1, 256});
        Tensor mid(ws.temb_mid, DType::BF16, {1, H});
        Tensor temb(ws.temb, DType::BF16, {1, H});
        linear(sin_in, w.time_embed_linear_1, mid, stream);
        dit_silu(mid, mid, stream);
        linear(mid, w.time_embed_linear_2, temb, stream);
    }

    // 7. mod = linear(silu(temb), modulation)  [1, 4H]
    Tensor temb(ws.temb, DType::BF16, {1, H});
    Tensor mod(ws.mod, DType::BF16, {1, 4 * H});
    dit_silu(temb, ws.temb_mid, stream);
    Tensor temb_silu(ws.temb_mid, DType::BF16, {1, H});
    linear(temb_silu, w.modulation, mod, stream);

    // RoPE freqs (precomputed on host, uploaded once per step; shape-dependent only).
    {
        // Build [S, 128] complex freqs (re/im interleaved) for the joint [text, image] sequence.
        const int Hh = (int)std::round(std::sqrt((double)N));  // image is Hh x Hh tokens
        std::vector<float> freqs((std::size_t)S * 128, 0.0F);
        std::vector<int> frame(S), height(S), width(S);
        for (int t = 0; t < M; t++) { frame[t] = t; height[t] = t; width[t] = t; }
        int hi = 0;
        for (int h = 0; h < Hh; h++) {
            for (int wd = 0; wd < Hh; wd++) {
                frame[M + hi]  = M;
                height[M + hi] = -(Hh - Hh / 2) + h;
                width[M + hi]  = -(Hh - Hh / 2) + wd;
                hi++;
            }
        }
        auto fill_axis = [&](int base_slot, int dim, const std::vector<int>& idx) {
            const int half = dim / 2;
            for (int t = 0; t < S; t++) {
                const float idxf = (float)idx[t];
                for (int i = 0; i < half; i++) {
                    const float angle = idxf * std::pow(10000.0F, -2.0F * i / dim);
                    freqs[(std::size_t)t * 128 + 2 * (base_slot + i)]     = std::cos(angle);
                    freqs[(std::size_t)t * 128 + 2 * (base_slot + i) + 1] = std::sin(angle);
                }
            }
        };
        fill_axis(0, 16, frame);    // 8 complex slots
        fill_axis(8, 56, height);   // 28 complex slots
        fill_axis(36, 56, width);   // 28 complex slots
        hipMemcpyAsync(ws.freqs, freqs.data(), freqs.size() * sizeof(float), hipMemcpyHostToDevice,
                       stream);
    }

    // 8. 32 layers
    Tensor q(ws.q, DType::BF16, {1, S, 32, DitConfig::head_dim});
    Tensor k(ws.k, DType::BF16, {1, S, 32, DitConfig::head_dim});
    Tensor v(ws.v, DType::BF16, {1, S, 32, DitConfig::head_dim});
    Tensor attn(ws.attn, DType::BF16, {1, S, H});
    Tensor x1(ws.x1, DType::BF16, {1, S, H});
    Tensor x2(ws.x2, DType::BF16, {1, S, H});
    Tensor mlp(ws.mlp, DType::BF16, {1, S, H});
    Tensor hs2(ws.hs2, DType::BF16, {1, S, H});

    for (int l = 0; l < DitConfig::layers; l++) {
        // attention sub-block
        dit_layer_norm(hs, DitConfig::eps, x1, stream);
        dit_modulate_scale(x1, Tensor(ws.mod, DType::BF16, {1, H}), x1, stream);  // scale1
        linear(x1, w.to_q[l], q, stream);
        linear(x1, w.to_k[l], k, stream);
        linear(x1, w.to_v[l], v, stream);
        rmsnorm(q, w.norm_q[l], DitConfig::eps, false, q, stream);
        rmsnorm(k, w.norm_k[l], DitConfig::eps, false, k, stream);
        dit_rope(ws.freqs, q, k, stream);
        dit_mha(q, k, v, attn, stream);
        linear(attn, w.to_out[l], attn, stream);
        dit_residual_gate(hs, Tensor(ws.mod + H, DType::BF16, {1, H}), attn, hs2, stream);  // gate1
        // mlp sub-block
        dit_layer_norm(hs2, DitConfig::eps, x2, stream);
        dit_modulate_scale(x2, Tensor(ws.mod + 2 * H, DType::BF16, {1, H}), x2, stream);  // scale2
        // swiglu: down(silu(gate(x2)) * up(x2))
        Tensor gate(ws.mlp, DType::BF16, {1, S, DitConfig::intermediate});
        Tensor up(ws.x1, DType::BF16, {1, S, DitConfig::intermediate});
        linear(x2, w.mlp_gate[l], gate, stream);
        linear(x2, w.mlp_up[l], up, stream);
        dit_silu(gate, gate, stream);
        // silu_mul(gate, up) -> mlp[1,S,intermediate]
        silu_mul(gate, up, Tensor(ws.mlp, DType::BF16, {1, S, DitConfig::intermediate}), stream);
        linear(Tensor(ws.mlp, DType::BF16, {1, S, DitConfig::intermediate}), w.mlp_down[l], mlp, stream);
        dit_residual_gate(hs2, Tensor(ws.mod + 3 * H, DType::BF16, {1, H}), mlp, hs, stream);  // gate2
    }

    // 9-10. norm_out: layernorm(hs) * (1 + scale)
    Tensor scale(ws.temb_mid, DType::BF16, {1, H});
    dit_silu(temb, ws.temb_mid, stream);
    linear(Tensor(ws.temb_mid, DType::BF16, {1, H}), w.norm_out_linear, scale, stream);
    dit_layer_norm(hs, DitConfig::eps, x1, stream);
    dit_modulate_scale(x1, scale, x1, stream);

    // 11. out = linear(x1, proj_out)  [1, S, C]
    Tensor out(ws.out, DType::BF16, {1, S, C});
    linear(x1, w.proj_out, out, stream);

    // 12. velocity = out[:, M:]  [1, N, C]
    hipMemcpyAsync(velocity.data(), ws.out + (std::size_t)M * C, (std::size_t)N * C * 2,
                   hipMemcpyDeviceToDevice, stream);
}

} // namespace ninfer::targets::qwen_image_21::detail
