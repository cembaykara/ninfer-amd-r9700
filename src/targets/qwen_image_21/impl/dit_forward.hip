// The Qwen-Image-2.1 DiT forward and denoise loop (wave32 gfx1201).
// Mirrors diffusers QwenImage21Transformer2DModel for text-to-image with causal_condition.
#include "dit_forward.h"

#include "core/device.h"

#include "dit_mha.h"
#include "dit_ops.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"

#include <hip/hip_bfloat16.h>

#include <cmath>
#include <fstream>
#include <vector>

namespace ninfer::targets::qwen_image_21::detail {
using namespace ninfer::ops;

namespace {

constexpr int H = DitConfig::hidden;
constexpr int I = DitConfig::intermediate;
constexpr int C = DitConfig::in_channels;

float bf16_round(float v) { return static_cast<float>(hip_bfloat16(v)); }

// [S][64] (cos, sin). Text advances one position on all three axes. Each image block freezes the
// frame axis at the position reached by the preceding text and lays its tokens on a height/width
// grid centred on zero. The condition block freezes at M; the target block freezes at
// M + max(cond_h, cond_w) (the reference advances the frame by max(h, w) per block).
void upload_rope(float* dst, int M, int a, int V, int cond_h, int cond_w, int grid_h, int grid_w,
                 hipStream_t stream) {
    const int Nc = cond_h * cond_w;
    const int Nt = grid_h * grid_w;
    const int b  = M - a - V;
    const int S  = a + Nc + b + Nt;
    // Reference layout [text_before][cond][text_after][target]: the condition block sits at the
    // VLM image slot (position a), the trailing text resumes after it, and the frame axis
    // advances by max(h, w) per image block.
    const int cond_start   = a;
    const int text2_start  = a + Nc;
    const int target_start = a + Nc + b;
    std::vector<int> frame(S), height(S), width(S);
    for (int t = 0; t < a; t++) { frame[t] = t; height[t] = t; width[t] = t; }
    for (int y = 0; y < cond_h; y++) {
        for (int x = 0; x < cond_w; x++) {
            const int t = cond_start + y * cond_w + x;
            frame[t]  = a;
            height[t] = y - (cond_h - cond_h / 2);
            width[t]  = x - (cond_w - cond_w / 2);
        }
    }
    for (int t = 0; t < b; t++) {
        const int pos = text2_start + t;
        const int f   = a + std::max(cond_h, cond_w) + t;
        frame[pos]  = f;
        height[pos] = f;
        width[pos]  = f;
    }
    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            const int t = target_start + y * grid_w + x;
            frame[t]  = a + std::max(cond_h, cond_w) + b;
            height[t] = y - (grid_h - grid_h / 2);
            width[t]  = x - (grid_w - grid_w / 2);
        }
    }
    std::vector<float> freqs((std::size_t)S * 128);
    const int axes[3] = {DitConfig::rope_axes[0], DitConfig::rope_axes[1], DitConfig::rope_axes[2]};
    const std::vector<int>* index[3] = {&frame, &height, &width};
    for (int t = 0; t < S; t++) {
        int slot = 0;
        for (int ax = 0; ax < 3; ax++) {
            for (int i = 0; i < axes[ax] / 2; i++, slot++) {
                const float inv = 1.0F / std::pow((float)DitConfig::rope_theta, (float)(2 * i) / axes[ax]);
                const float angle = (float)(*index[ax])[t] * inv;
                freqs[(std::size_t)t * 128 + 2 * slot]     = std::cos(angle);
                freqs[(std::size_t)t * 128 + 2 * slot + 1] = std::sin(angle);
            }
        }
    }
    HIP_CHECK(hipMemcpyAsync(dst, freqs.data(), freqs.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}

// Column 0: the sampled timestep; column 1: t=0. The reference rounds sigma*1000 and the /1000
// back to BF16 before the sinusoid, so the same rounding is applied here.
void upload_timestep(hip_bfloat16* dst, float sigma, hipStream_t stream) {
    const float t = bf16_round(bf16_round(sigma * 1000.0F) / 1000.0F) * 1000.0F;
    std::vector<hip_bfloat16> emb(2 * 256);
    for (int i = 0; i < 128; i++) {
        const float freq = std::exp(-std::log(10000.0F) * (float)i / 128.0F);
        emb[i]             = hip_bfloat16(std::cos(t * freq));
        emb[128 + i]       = hip_bfloat16(std::sin(t * freq));
        emb[256 + i]       = hip_bfloat16(1.0F);
        emb[256 + 128 + i] = hip_bfloat16(0.0F);
    }
    HIP_CHECK(hipMemcpyAsync(dst, emb.data(), emb.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}

void forward(const DitWeights& w, const Tensor& latent, const Tensor& cond_latent, float sigma, int M, int a,
             int V, int Nc, int Nt, DitWorkspace& ws, Tensor& velocity, hipStream_t stream) {
    const int b = M - a - V;
    // Reference layout [text_before(a)][cond(Nc)][text_after(b)][target(Nt)]: the condition block
    // sits at the VLM image slot, so the text stream is split around it. The V image-pad tokens
    // are dropped (the cond block replaces them), so the joint is a + Nc + b + Nt, not M + Nc + Nt.
    // The t=0/sampled modulation split and the output slice are at the target start a + Nc + b.
    const int S = a + Nc + b + Nt;
    if (std::getenv("NINFER_DIT_TXT_DUMP") != nullptr) {
        HIP_CHECK(hipStreamSynchronize(stream));
        std::ofstream tf(std::getenv("NINFER_DIT_TXT_DUMP"), std::ios::binary);
        tf.write(reinterpret_cast<const char*>(ws.txt),
                 (std::streamsize)(std::size_t)H * M * 2);
    }
    if (std::getenv("NINFER_DIT_TARGET_DUMP") != nullptr) {
        HIP_CHECK(hipStreamSynchronize(stream));
        std::ofstream df(std::getenv("NINFER_DIT_TARGET_DUMP"), std::ios::binary);
        df.write(reinterpret_cast<const char*>(latent.data),
                 (std::streamsize)(std::size_t)64 * Nt * 2);
    }
    if (Nc > 0 && std::getenv("NINFER_DIT_PACK_DUMP") != nullptr) {
        HIP_CHECK(hipStreamSynchronize(stream));
        std::ofstream pf(std::getenv("NINFER_DIT_PACK_DUMP"), std::ios::binary);
        pf.write(reinterpret_cast<const char*>(cond_latent.data),
                 (std::streamsize)(std::size_t)64 * Nc * 2);
    }

    Tensor hs(ws.hs, DType::BF16, {H, S});
    if (a > 0) {
        HIP_CHECK(hipMemcpyAsync(ws.hs, ws.txt, (std::size_t)a * H * 2, hipMemcpyDeviceToDevice, stream));
    }
    if (Nc > 0) {
        Tensor cond(ws.hs + (std::size_t)a * H, DType::BF16, {H, Nc});
        dit_linear(cond_latent, w.img_in, cond, stream);
        if (std::getenv("NINFER_DIT_EMB_DUMP") != nullptr) {
        HIP_CHECK(hipStreamSynchronize(stream));
        std::ofstream ef(std::getenv("NINFER_DIT_EMB_DUMP"), std::ios::binary);
        ef.write(reinterpret_cast<const char*>(cond.data),
                 (std::streamsize)(std::size_t)H * Nc * 2);
    }
    }
    if (b > 0) {
        HIP_CHECK(hipMemcpyAsync(ws.hs + (std::size_t)(a + Nc) * H, ws.txt + (std::size_t)(a + V) * H,
                                  (std::size_t)b * H * 2, hipMemcpyDeviceToDevice, stream));
    }
    Tensor img(ws.hs + (std::size_t)(a + Nc + b) * H, DType::BF16, {H, Nt});
    dit_linear(latent, w.img_in, img, stream);

    upload_timestep(ws.t_sin, sigma, stream);
    Tensor t_sin(ws.t_sin, DType::BF16, {256, 2});
    Tensor t_mid(ws.t_mid, DType::BF16, {H, 2});
    Tensor temb(ws.temb, DType::BF16, {H, 2});
    Tensor temb_silu(ws.temb_silu, DType::BF16, {H, 2});
    Tensor mod(ws.mod, DType::BF16, {4 * H, 2});
    Tensor out_scale(ws.out_scale, DType::BF16, {H, 2});
    dit_linear(t_sin, w.time_embed_linear_1, t_mid, stream);
    dit_silu(t_mid, t_mid, stream);
    dit_linear(t_mid, w.time_embed_linear_2, temb, stream);
    dit_silu(temb, temb_silu, stream);
    dit_linear(temb_silu, w.modulation, mod, stream);
    dit_linear(temb_silu, w.norm_out_linear, out_scale, stream);
    const hip_bfloat16* mod_img = ws.mod;
    const hip_bfloat16* mod_txt = ws.mod + 4 * H;

    Tensor x1(ws.x1, DType::BF16, {H, S});
    Tensor x2(ws.x2, DType::BF16, {H, S});
    Tensor q(ws.q, DType::BF16, {H, S});
    Tensor k(ws.k, DType::BF16, {H, S});
    Tensor v(ws.v, DType::BF16, {H, S});
    Tensor qn(ws.qn, DType::BF16, {DitConfig::head_dim, DitConfig::heads, S});
    Tensor kn(ws.kn, DType::BF16, {DitConfig::head_dim, DitConfig::heads, S});
    Tensor attn(ws.attn, DType::BF16, {H, S});
    Tensor gate(ws.gate, DType::BF16, {I, S});
    Tensor up(ws.up, DType::BF16, {I, S});
    Tensor act(ws.act, DType::BF16, {I, S});

    for (int l = 0; l < DitConfig::layers; l++) {
        if (l == 0 && std::getenv("NINFER_DIT_PRE0_DUMP") != nullptr) {
        HIP_CHECK(hipStreamSynchronize(stream));
        std::ofstream hf(std::getenv("NINFER_DIT_PRE0_DUMP"), std::ios::binary);
        hf.write(reinterpret_cast<const char*>(ws.hs),
                 (std::streamsize)(std::size_t)H * S * 2);
    }
        dit_layer_norm(hs, DitConfig::eps, x1, stream);
        dit_modulate(x1, mod_img + DitConfig::modulation_scale1, mod_txt + DitConfig::modulation_scale1, a + Nc + b,
                     stream);
        dit_linear(x1, w.to_q[l], q, stream);
        dit_linear(x1, w.to_k[l], k, stream);
        dit_linear(x1, w.to_v[l], v, stream);
        rmsnorm(q.view({DitConfig::head_dim, DitConfig::heads, S}), w.norm_q[l], DitConfig::eps, false, qn,
                stream);
        rmsnorm(k.view({DitConfig::head_dim, DitConfig::heads, S}), w.norm_k[l], DitConfig::eps, false, kn,
                stream);
        dit_rope(ws.freqs, qn, kn, stream);
        if (l == 0 && std::getenv("NINFER_DIT_QN_DUMP") != nullptr) {
            HIP_CHECK(hipStreamSynchronize(stream));
            std::ofstream qf(std::getenv("NINFER_DIT_QN_DUMP"), std::ios::binary);
            qf.write(reinterpret_cast<const char*>(qn.data),
                     (std::streamsize)(std::size_t)DitConfig::head_dim * DitConfig::heads * S * 2);
        }
        Tensor attn_heads = attn.view({DitConfig::head_dim, DitConfig::heads, S});
        dit_mha(qn, kn, v.view({DitConfig::head_dim, DitConfig::heads, S}), a, Nc, b, attn_heads, stream);
        dit_linear(attn, w.to_out[l], x2, stream);
        dit_residual_gate(hs, mod_img + DitConfig::modulation_gate1, mod_txt + DitConfig::modulation_gate1, a + Nc + b,
                          x2, stream);

        dit_layer_norm(hs, DitConfig::eps, x1, stream);
        dit_modulate(x1, mod_img + DitConfig::modulation_scale2, mod_txt + DitConfig::modulation_scale2, a + Nc + b,
                     stream);
        dit_linear(x1, w.mlp_gate[l], gate, stream);
        dit_linear(x1, w.mlp_up[l], up, stream);
        silu_mul(gate, up, act, stream);
        dit_linear(act, w.mlp_down[l], x2, stream);
        dit_residual_gate(hs, mod_img + DitConfig::modulation_gate2, mod_txt + DitConfig::modulation_gate2, a + Nc + b,
                          x2, stream);
        if (std::getenv("NINFER_DIT_LAYER_DUMP") != nullptr) {
            HIP_CHECK(hipStreamSynchronize(stream));
            std::string lf = std::string(std::getenv("NINFER_DIT_LAYER_DUMP")) + "_" + std::to_string(l) + ".bin";
            std::ofstream lfs(lf, std::ios::binary);
            lfs.write(reinterpret_cast<const char*>(ws.hs),
                      (std::streamsize)(std::size_t)H * S * 2);
        }
    }

    dit_layer_norm(hs, DitConfig::eps, x1, stream);
    dit_modulate(x1, ws.out_scale, ws.out_scale + H, a + Nc + b, stream);
    const Tensor x1_img(ws.x1 + (std::size_t)(a + Nc + b) * H, DType::BF16, {H, Nt});
    dit_linear(x1_img, w.proj_out, velocity, stream);
}

} // namespace

void dit_denoise(const DitWeights& w, const Tensor& text_emb, int text_split, int vlm_img_tokens,
                 int cond_h, int cond_w, int grid_h, int grid_w,
                 const Tensor& cond_latent, Tensor& latent, DitWorkspace& ws, hipStream_t stream) {
    const int M = (int)text_emb.ne[1];
    const int Nc = cond_h * cond_w;
    const int Nt = grid_h * grid_w;

    // The text stream does not depend on the timestep: project it once.
    Tensor x1(ws.x1, DType::BF16, {H, M});
    Tensor x2(ws.x2, DType::BF16, {H, M});
    Tensor txt(ws.txt, DType::BF16, {H, M});
    rmsnorm(text_emb, w.txt_in_text_norm, DitConfig::eps, true, x1, stream);
    dit_linear(x1, w.txt_in_in_layer, x2, stream);
    gelu(x2, GeluMode::Tanh, stream);
    dit_linear(x2, w.txt_in_out_layer, txt, stream);

    upload_rope(ws.freqs, M, text_split, vlm_img_tokens, cond_h, cond_w, grid_h, grid_w, stream);

    Tensor velocity(ws.velocity, DType::BF16, {C, Nt});
    const auto& sigmas = DitConfig::sample_sigmas;
    constexpr int steps = sizeof(DitConfig::sample_sigmas) / sizeof(float);
    // Causal conditioning: FULL sigmas (1.0 -> 0) for both T2I and image-conditioned generation.
    // The condition latent is a fixed prefix (not updated); only the target latent is denoised.
    for (int i = 0; i < steps; i++) {
        const float cur  = sigmas[i];
        const float next = (i + 1 < steps ? sigmas[i + 1] : 0.0F);
        forward(w, latent, cond_latent, cur, M, text_split, vlm_img_tokens, Nc, Nt, ws, velocity, stream);
        dit_euler(latent, velocity, next - cur, stream);
    }
}

} // namespace ninfer::targets::qwen_image_21::detail
