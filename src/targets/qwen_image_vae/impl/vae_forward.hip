// VAE decode forward (single image, 2D): latent [1,64,H/16,W/16] -> image [1,4,H,W].
// Decoder: conv_in -> mid(2 resnet + attn) -> 5 up_blocks -> RMS -> SiLU -> conv_out.
// For a single frame the temporal dim stays 1 (no temporal upsampling).
#include "vae_forward.h"

#include "ninfer/ops/conv2d_3x3.h"
#include "ninfer/ops/upsample2d_nearest.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace ninfer::targets::qwen_image_vae {
namespace detail {

constexpr int kLatentCh = 64;
constexpr int kImageCh  = 4;
constexpr int kMidDim   = 1152;
constexpr int kUpOut[5] = {1152, 1152, 576, 288, 144};
constexpr int kUpIn[5]  = {1152, 1152, 1152, 576, 288};

// Tensor helpers (NCHW, BF16)
inline Tensor t3(const void* d, int c, int h, int w) {
    Tensor t; t.data = const_cast<void*>(d); t.dtype = DType::BF16;
    t.ne[0] = c; t.ne[1] = h; t.ne[2] = w; t.ne[3] = 1;
    t.nb[0] = 2; t.nb[1] = (std::int64_t)c * w * 2; t.nb[2] = (std::int64_t)c * h * w * 2; t.nb[3] = 0;
    return t;
}
inline Tensor t4(const void* d, int a, int b, int c, int e) {
    Tensor t; t.data = const_cast<void*>(d); t.dtype = DType::BF16;
    t.ne[0] = a; t.ne[1] = b; t.ne[2] = c; t.ne[3] = e;
    t.nb[0] = 2; t.nb[1] = (std::int64_t)a * 2; t.nb[2] = (std::int64_t)a * b * 2; t.nb[3] = (std::int64_t)a * b * c * 2;
    return t;
}
inline Tensor t1(const void* d, int n) {
    Tensor t; t.data = const_cast<void*>(d); t.dtype = DType::BF16;
    t.ne[0] = n; t.ne[1] = t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 2; t.nb[1] = t.nb[2] = t.nb[3] = (std::int64_t)n * 2;
    return t;
}

// RMS norm over channels: out = (x / ||x||_2) * sqrt(C) * gamma, per (h,w)
__global__ void rms_channels_kernel(const hip_bfloat16* __restrict__ in,
                                    const hip_bfloat16* __restrict__ gamma,
                                    int c, int h, int w, hip_bfloat16* __restrict__ out) {
    const int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= h * w) return;
    const int y = hw / w, x = hw % w;
    float ss = 0.0f;
    for (int ci = 0; ci < c; ++ci) {
        float v = float(in[(std::int64_t)ci * h * w + (std::int64_t)y * w + x]);
        ss += v * v;
    }
    const float inv = 1.0f / sqrtf(ss + 1e-6f);
    const float scale = sqrtf((float)c) * inv;
    for (int ci = 0; ci < c; ++ci) {
        float v = float(in[(std::int64_t)ci * h * w + (std::int64_t)y * w + x]);
        out[(std::int64_t)ci * h * w + (std::int64_t)y * w + x] =
            hip_bfloat16(v * scale * float(gamma[ci]));
    }
}
void rms_channels(const hip_bfloat16* in, const hip_bfloat16* g, int c, int h, int w,
                  hip_bfloat16* out, hipStream_t s) {
    dim3 blk(256); dim3 grd((h * w + 255) / 256);
    rms_channels_kernel<<<grd, blk, 0, s>>>(in, g, c, h, w, out);
}

// SiLU
__global__ void silu_kernel(const hip_bfloat16* __restrict__ in, std::int64_t n, hip_bfloat16* __restrict__ out) {
    std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = float(in[i]);
    out[i] = hip_bfloat16(x / (1.0f + expf(-x)));
}
void silu(const hip_bfloat16* in, std::int64_t n, hip_bfloat16* out, hipStream_t s) {
    dim3 blk(256); dim3 grd((n + 255) / 256);
    silu_kernel<<<grd, blk, 0, s>>>(in, n, out);
}

// element-wise add
__global__ void add_kernel(const hip_bfloat16* __restrict__ a, const hip_bfloat16* __restrict__ b,
                           std::int64_t n, hip_bfloat16* __restrict__ out) {
    std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = hip_bfloat16(float(a[i]) + float(b[i]));
}
void add(const hip_bfloat16* a, const hip_bfloat16* b, std::int64_t n, hip_bfloat16* out, hipStream_t s) {
    dim3 blk(256); dim3 grd((n + 255) / 256);
    add_kernel<<<grd, blk, 0, s>>>(a, b, n, out);
}

// 1x1 conv (channel mix): out[cout,hw] = sum_c in[c,hw]*W[cout,c] + b[cout]
__global__ void conv1x1_kernel(const hip_bfloat16* __restrict__ in, const hip_bfloat16* __restrict__ w,
                               const hip_bfloat16* __restrict__ b, int cin, int cout, int hw,
                               hip_bfloat16* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cout * hw) return;
    const int co = idx / hw, p = idx % hw;
    float acc = float(b[co]);
    for (int ci = 0; ci < cin; ++ci)
        acc += float(w[(std::int64_t)co * cin + ci]) * float(in[(std::int64_t)ci * hw + p]);
    out[idx] = hip_bfloat16(acc);
}
void conv1x1(const hip_bfloat16* in, const hip_bfloat16* w, const hip_bfloat16* b,
             int cin, int cout, int hw, hip_bfloat16* out, hipStream_t s) {
    dim3 blk(256); dim3 grd((std::int64_t)cout * hw / 256 + 1);
    conv1x1_kernel<<<grd, blk, 0, s>>>(in, w, b, cin, cout, hw, out);
}

// Residual block
void resnet(const VaeWeights::Resnet& r, const hip_bfloat16* in, int cin, int cout, int h, int w,
            hip_bfloat16* out, VaeWorkspace& ws, hipStream_t s) {
    const std::int64_t n_in = (std::int64_t)cin * h * w, n_out = (std::int64_t)cout * h * w;
    const int hw = h * w;
    if (r.shortcut) conv1x1(in, r.shortcut, r.shortcut_b, cin, cout, hw, ws.a, s);
    rms_channels(in, r.norm1, cin, h, w, ws.b, s);
    silu(ws.b, n_in, ws.c, s);
    Tensor in_t = t3(ws.c, cin, h, w), w1 = t4(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(r.conv1)), cout, cin, 3, 3),
           b1 = t1(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(r.conv1_b)), cout), out_t = t3(ws.d, cout, h, w);
    ops::conv2d_3x3(in_t, w1, b1, out_t, s);
    rms_channels(ws.d, r.norm2, cout, h, w, ws.e, s);
    silu(ws.e, n_out, ws.f, s);
    Tensor in2 = t3(ws.f, cout, h, w), w2 = t4(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(r.conv2)), cout, cout, 3, 3),
           b2 = t1(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(r.conv2_b)), cout), out2 = t3(ws.g, cout, h, w);
    ops::conv2d_3x3(in2, w2, b2, out2, s);
    if (r.shortcut) add(ws.g, ws.a, n_out, out, s);
    else add(ws.g, in, n_out, out, s);
}

// Upsampler (single image): nearest 2x spatial + resample conv 3x3
void upsample(const VaeWeights::Upsampler& u, const hip_bfloat16* in, int c, int h, int w,
              hip_bfloat16* out, VaeWorkspace& ws, hipStream_t s) {
    Tensor in_t = t3(in, c, h, w), up_t = t3(ws.a, c, 2 * h, 2 * w);
    ops::upsample2d_nearest(in_t, up_t, s);
    Tensor in2 = t3(ws.a, c, 2 * h, 2 * w),
           w1 = t4(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(u.resample)), c, c, 3, 3),
           b1 = t1(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(u.resample_b)), c),
           out_t = t3(out, c, 2 * h, 2 * w);
    ops::conv2d_3x3(in2, w1, b1, out_t, s);
}

// Single-head self-attention over spatial positions (head_dim = dim).
// q,k,v: [dim, hw] (from to_qkv). attn: [hw, hw] weights. out: [dim, hw].
__global__ void attn_row_kernel(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
                                int dim, int hw, hip_bfloat16* __restrict__ attn) {
    const int qpos = blockIdx.x;
    if (qpos >= hw) return;
    const float scale = 1.0f / sqrtf((float)dim);
    // pass 1: max
    float mx = -1e30f;
    for (int kp = 0; kp < hw; ++kp) {
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) dot += float(q[(std::int64_t)d * hw + qpos]) * float(k[(std::int64_t)d * hw + kp]);
        mx = fmaxf(mx, dot * scale);
    }
    // pass 2: unnormalized exp + sum
    float sum = 0.0f;
    for (int kp = 0; kp < hw; ++kp) {
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) dot += float(q[(std::int64_t)d * hw + qpos]) * float(k[(std::int64_t)d * hw + kp]);
        float e = expf(dot * scale - mx);
        attn[(std::int64_t)qpos * hw + kp] = hip_bfloat16(e);
        sum += e;
    }
    // pass 3: normalize
    const float inv = 1.0f / sum;
    for (int kp = 0; kp < hw; ++kp)
        attn[(std::int64_t)qpos * hw + kp] = hip_bfloat16(float(attn[(std::int64_t)qpos * hw + kp]) * inv);
}
__global__ void attn_out_kernel(const hip_bfloat16* __restrict__ attn, const hip_bfloat16* __restrict__ v,
                                int dim, int hw, hip_bfloat16* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * hw) return;
    const int c = idx / hw, qpos = idx % hw;
    float acc = 0.0f;
    for (int kp = 0; kp < hw; ++kp)
        acc += float(attn[(std::int64_t)qpos * hw + kp]) * float(v[(std::int64_t)c * hw + kp]);
    out[idx] = hip_bfloat16(acc);
}
void mid_attention(const VaeWeights& wts, const hip_bfloat16* in, int dim, int h, int w,
                   hip_bfloat16* out, VaeWorkspace& ws, hipStream_t s) {
    const int hw = h * w;
    rms_channels(in, wts.mid_attn_norm, dim, h, w, ws.a, s);
    conv1x1(ws.a, wts.mid_attn_to_qkv, wts.mid_attn_to_qkv_b, dim, 3 * dim, hw, ws.b, s);
    // q = ws.b[0:dim], k = ws.b[dim:2dim], v = ws.b[2dim:3dim]
    const hip_bfloat16* q = ws.b;
    const hip_bfloat16* k = ws.b + (std::int64_t)dim * hw;
    const hip_bfloat16* v = ws.b + (std::int64_t)2 * dim * hw;
    dim3 blk(256);
    attn_row_kernel<<<hw, blk, 0, s>>>(q, k, dim, hw, ws.c);
    dim3 grd((std::int64_t)dim * hw / 256 + 1);
    attn_out_kernel<<<grd, blk, 0, s>>>(ws.c, v, dim, hw, ws.d);
    // proj: [dim, hw] -> [dim, hw]
    conv1x1(ws.d, wts.mid_attn_proj, wts.mid_attn_proj_b, dim, dim, hw, out, s);
}

} // namespace detail

namespace detail {

void vae_decode(const VaeWeights& w, const Tensor& latent, Tensor& out_image,
                VaeWorkspace& ws, hipStream_t s) {
    const int h0 = latent.ne[2], w0 = latent.ne[3];

    Tensor lat = t3(latent.data, kLatentCh, h0, w0),
           win = t4(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(w.conv_in)), kMidDim, kLatentCh, 3, 3),
           bin = t1(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(w.conv_in_b)), kMidDim),
           a0 = t3(ws.a, kMidDim, h0, w0);
    ops::conv2d_3x3(lat, win, bin, a0, s);

    resnet(w.mid_resnets[0], ws.a, kMidDim, kMidDim, h0, w0, ws.b, ws, s);
    resnet(w.mid_resnets[1], ws.b, kMidDim, kMidDim, h0, w0, ws.c, ws, s);
    mid_attention(w, ws.c, kMidDim, h0, w0, ws.d, ws, s);

    hip_bfloat16* cur = ws.d; int ch = kMidDim; int hh = h0, ww = w0;
    for (int i = 0; i < 5; ++i) {
        resnet(w.up_blocks[i].resnets[0], cur, kUpIn[i], kUpOut[i], hh, ww, ws.a, ws, s);
        resnet(w.up_blocks[i].resnets[1], ws.a, kUpOut[i], kUpOut[i], hh, ww, ws.b, ws, s);
        resnet(w.up_blocks[i].resnets[2], ws.b, kUpOut[i], kUpOut[i], hh, ww, ws.c, ws, s);
        cur = ws.c; ch = kUpOut[i];
        if (i < 4) {
            upsample(w.up_blocks[i].upsampler, cur, ch, hh, ww, ws.e, ws, s);
            cur = ws.e; hh *= 2; ww *= 2;
        }
    }

    rms_channels(cur, w.norm_out, ch, hh, ww, ws.f, s);
    silu(ws.f, (std::int64_t)ch * hh * ww, ws.g, s);
    Tensor gin = t3(ws.g, ch, hh, ww),
           wout = t4(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(w.conv_out)), kImageCh, ch, 3, 3),
           bout = t1(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(w.conv_out_b)), kImageCh),
           img = t3(out_image.data, kImageCh, hh, ww);
    ops::conv2d_3x3(gin, wout, bout, img, s);
}

} // namespace detail
} // namespace ninfer::targets::qwen_image_vae
