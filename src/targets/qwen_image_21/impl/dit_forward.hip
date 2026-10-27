// The Qwen-Image-2.1 DiT forward and denoise loop (wave32 gfx1201).
// Mirrors diffusers QwenImage21Transformer2DModel with causal_condition.
// Multi-ref joint: [text][cond0][text][cond1]…[text][target] (HF img_mask interleave).
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
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen_image_21::detail {
using namespace ninfer::ops;

namespace {

constexpr int H = DitConfig::hidden;
constexpr int I = DitConfig::intermediate;
constexpr int C = DitConfig::in_channels;
constexpr int kMaxCond = 10;

float bf16_round(float v) { return static_cast<float>(hip_bfloat16(v)); }

void upload_rope(float* dst, int S, int prefix_len, const int* img_starts, const int* img_ends,
                 int n_img, const int* cond_hs, const int* cond_ws, int grid_h, int grid_w,
                 hipStream_t stream) {
    std::vector<int> frame(S), height(S), width(S);
    int joint = 0;
    int frame_cursor = 0;
    auto emit_text = [&](int n) {
        for (int t = 0; t < n; ++t) {
            frame[joint] = height[joint] = width[joint] = frame_cursor;
            ++frame_cursor;
            ++joint;
        }
    };
    auto emit_cond = [&](int i) {
        const int ch = cond_hs[i];
        const int cw = cond_ws[i];
        const int frame_base = frame_cursor;
        for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
                frame[joint]  = frame_base;
                height[joint] = y - (ch - ch / 2);
                width[joint]  = x - (cw - cw / 2);
                ++joint;
            }
        }
        frame_cursor += std::max(ch, cw);
    };
    int prev = 0;
    for (int i = 0; i < n_img; ++i) {
        emit_text(img_starts[i] - prev);
        emit_cond(i);
        prev = img_ends[i];
    }
    emit_text(prefix_len - prev);
    const int frame_base = frame_cursor;
    for (int y = 0; y < grid_h; ++y) {
        for (int x = 0; x < grid_w; ++x) {
            frame[joint]  = frame_base;
            height[joint] = y - (grid_h - grid_h / 2);
            width[joint]  = x - (grid_w - grid_w / 2);
            ++joint;
        }
    }
    if (joint != S) { throw std::runtime_error("dit rope: joint length mismatch"); }

    std::vector<float> freqs((std::size_t)S * 128);
    const int axes[3] = {DitConfig::rope_axes[0], DitConfig::rope_axes[1], DitConfig::rope_axes[2]};
    const std::vector<int>* index[3] = {&frame, &height, &width};
    for (int t = 0; t < S; t++) {
        int slot = 0;
        for (int ax = 0; ax < 3; ax++) {
            for (int i = 0; i < axes[ax] / 2; i++, slot++) {
                const float inv =
                    1.0F / std::pow((float)DitConfig::rope_theta, (float)(2 * i) / axes[ax]);
                const float angle = (float)(*index[ax])[t] * inv;
                freqs[(std::size_t)t * 128 + 2 * slot]     = std::cos(angle);
                freqs[(std::size_t)t * 128 + 2 * slot + 1] = std::sin(angle);
            }
        }
    }
    HIP_CHECK(hipMemcpyAsync(dst, freqs.data(), freqs.size() * sizeof(float), hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}

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
    HIP_CHECK(hipMemcpyAsync(dst, emb.data(), emb.size() * sizeof(hip_bfloat16),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}

void forward(const DitWeights& w, const Tensor& latent, const Tensor& cond_latent, float sigma,
             int M, const int* pad_starts, const int* pad_counts, const int* cond_hs,
             const int* cond_ws, int n_cond, int grid_h, int grid_w, DitWorkspace& ws,
             Tensor& velocity, hipStream_t stream) {
    if (n_cond < 0 || n_cond > kMaxCond) {
        throw std::runtime_error("dit forward: n_cond out of range");
    }
    const int Nt = grid_h * grid_w;
    int Nc = 0;
    int V_pads = 0;
    for (int i = 0; i < n_cond; ++i) {
        if (cond_hs[i] <= 0 || cond_ws[i] <= 0 || pad_counts[i] <= 0) {
            throw std::runtime_error("dit forward: invalid cond/pad shape");
        }
        Nc += cond_hs[i] * cond_ws[i];
        V_pads += pad_counts[i];
    }
    const int S = M - V_pads + Nc + Nt;
    const int prefix_len = S - Nt;

    int img_starts[kMaxCond];
    int img_ends[kMaxCond];
    int joint = 0;
    int text_cursor = 0;
    int cond_off = 0;
    for (int i = 0; i < n_cond; ++i) {
        if (pad_starts[i] < text_cursor || pad_starts[i] + pad_counts[i] > M) {
            throw std::runtime_error("dit forward: pad run out of order/range");
        }
        const int gap = pad_starts[i] - text_cursor;
        if (gap > 0) {
            HIP_CHECK(hipMemcpyAsync(ws.hs + (std::size_t)joint * H,
                                     ws.txt + (std::size_t)text_cursor * H,
                                     (std::size_t)gap * H * 2, hipMemcpyDeviceToDevice, stream));
            joint += gap;
        }
        const int nci = cond_hs[i] * cond_ws[i];
        img_starts[i] = joint;
        Tensor cond_src = cond_latent.slice(1, cond_off, nci);
        Tensor cond_dst(ws.hs + (std::size_t)joint * H, DType::BF16, {H, nci});
        dit_linear(cond_src, w.img_in, cond_dst, stream);
        img_ends[i] = joint + nci;
        joint += nci;
        cond_off += nci;
        text_cursor = pad_starts[i] + pad_counts[i];
    }
    const int tail = M - text_cursor;
    if (tail > 0) {
        HIP_CHECK(hipMemcpyAsync(ws.hs + (std::size_t)joint * H,
                                 ws.txt + (std::size_t)text_cursor * H, (std::size_t)tail * H * 2,
                                 hipMemcpyDeviceToDevice, stream));
        joint += tail;
    }
    if (joint != prefix_len) {
        throw std::runtime_error("dit forward: prefix length mismatch");
    }
    Tensor hs(ws.hs, DType::BF16, {H, S});
    Tensor img(ws.hs + (std::size_t)prefix_len * H, DType::BF16, {H, Nt});
    dit_linear(latent, w.img_in, img, stream);

    upload_rope(ws.freqs, S, prefix_len, img_starts, img_ends, n_cond, cond_hs, cond_ws, grid_h,
                grid_w, stream);

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
        dit_layer_norm(hs, DitConfig::eps, x1, stream);
        dit_modulate(x1, mod_img + DitConfig::modulation_scale1,
                     mod_txt + DitConfig::modulation_scale1, prefix_len, stream);
        dit_linear(x1, w.to_q[l], q, stream);
        dit_linear(x1, w.to_k[l], k, stream);
        dit_linear(x1, w.to_v[l], v, stream);
        rmsnorm(q.view({DitConfig::head_dim, DitConfig::heads, S}), w.norm_q[l], DitConfig::eps,
                false, qn, stream);
        rmsnorm(k.view({DitConfig::head_dim, DitConfig::heads, S}), w.norm_k[l], DitConfig::eps,
                false, kn, stream);
        dit_rope(ws.freqs, qn, kn, stream);
        Tensor attn_heads = attn.view({DitConfig::head_dim, DitConfig::heads, S});
        dit_mha(qn, kn, v.view({DitConfig::head_dim, DitConfig::heads, S}), img_starts, img_ends,
                n_cond, prefix_len, attn_heads, stream);
        dit_linear(attn, w.to_out[l], x2, stream);
        dit_residual_gate(hs, mod_img + DitConfig::modulation_gate1,
                          mod_txt + DitConfig::modulation_gate1, prefix_len, x2, stream);

        dit_layer_norm(hs, DitConfig::eps, x1, stream);
        dit_modulate(x1, mod_img + DitConfig::modulation_scale2,
                     mod_txt + DitConfig::modulation_scale2, prefix_len, stream);
        dit_linear(x1, w.mlp_gate[l], gate, stream);
        dit_linear(x1, w.mlp_up[l], up, stream);
        silu_mul(gate, up, act, stream);
        dit_linear(act, w.mlp_down[l], x2, stream);
        dit_residual_gate(hs, mod_img + DitConfig::modulation_gate2,
                          mod_txt + DitConfig::modulation_gate2, prefix_len, x2, stream);
    }

    dit_layer_norm(hs, DitConfig::eps, x1, stream);
    dit_modulate(x1, ws.out_scale, ws.out_scale + H, prefix_len, stream);
    const Tensor x1_img(ws.x1 + (std::size_t)prefix_len * H, DType::BF16, {H, Nt});
    dit_linear(x1_img, w.proj_out, velocity, stream);
}

} // namespace

void dit_denoise(const DitWeights& w, const Tensor& text_emb, const int* pad_starts,
                 const int* pad_counts, const int* cond_hs, const int* cond_ws, int n_cond,
                 int grid_h, int grid_w, const Tensor& cond_latent, Tensor& latent,
                 DitWorkspace& ws, hipStream_t stream, int step_begin) {
    const int M = (int)text_emb.ne[1];
    const int Nt = grid_h * grid_w;

    Tensor x1(ws.x1, DType::BF16, {H, M});
    Tensor x2(ws.x2, DType::BF16, {H, M});
    Tensor txt(ws.txt, DType::BF16, {H, M});
    rmsnorm(text_emb, w.txt_in_text_norm, DitConfig::eps, true, x1, stream);
    dit_linear(x1, w.txt_in_in_layer, x2, stream);
    gelu(x2, GeluMode::Tanh, stream);
    dit_linear(x2, w.txt_in_out_layer, txt, stream);

    Tensor velocity(ws.velocity, DType::BF16, {C, Nt});
    const auto& sigmas = DitConfig::sample_sigmas;
    constexpr int steps = sizeof(DitConfig::sample_sigmas) / sizeof(float);
    if (step_begin < 0) { step_begin = 0; }
    if (step_begin >= steps) { step_begin = steps - 1; }
    for (int i = step_begin; i < steps; i++) {
        const float cur  = sigmas[i];
        const float next = (i + 1 < steps ? sigmas[i + 1] : 0.0F);
        forward(w, latent, cond_latent, cur, M, pad_starts, pad_counts, cond_hs, cond_ws, n_cond,
                grid_h, grid_w, ws, velocity, stream);
        dit_euler(latent, velocity, next - cur, stream);
    }
}

} // namespace ninfer::targets::qwen_image_21::detail
