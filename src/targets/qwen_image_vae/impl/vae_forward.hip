// VAE decode forward (single image, 2D): latent [1,64,H/16,W/16] -> image [1,4,H,W].
// Decoder: conv_in -> mid(2 resnet + attn) -> 5 up_blocks -> RMS -> SiLU -> conv_out.
// For a single frame the temporal dim stays 1 (no temporal upsampling).
#include "vae_forward.h"

#include "core/device.h"
#include "ninfer/ops/conv2d_3x3.h"
#include "ninfer/ops/upsample2d_nearest.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <stdexcept>

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

__global__ void copy_kernel(const hip_bfloat16* __restrict__ in, std::int64_t n, hip_bfloat16* __restrict__ out) {
    const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}
void copy_bf16(const hip_bfloat16* in, std::int64_t n, hip_bfloat16* out, hipStream_t s) {
    copy_kernel<<<(n + 255) / 256, 256, 0, s>>>(in, n, out);
}

// Residual block
void resnet(const VaeWeights::Resnet& r, const hip_bfloat16* in, int cin, int cout, int h, int w,
            hip_bfloat16* out, VaeWorkspace& ws, hipStream_t s) {
    const std::int64_t n_in = (std::int64_t)cin * h * w, n_out = (std::int64_t)cout * h * w;
    const int hw = h * w;
    // The conv temps reuse ws.b..ws.g, so the residual has to live in ws.a.
    const hip_bfloat16* skip = in;
    if (r.shortcut) {
        conv1x1(in, r.shortcut, r.shortcut_b, cin, cout, hw, ws.a, s);
        skip = ws.a;
    } else if (in != ws.a) {
        copy_bf16(in, n_in, ws.a, s);
        skip = ws.a;
    }
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
    add(ws.g, skip, n_out, out, s);
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
    // Single-head attention over spatial positions, plus the input residual.
    // `in` must not alias ws.a/ws.e/ws.f/ws.g. qkv is [3*dim, hw] in ws.e.
    const int hw = h * w;
    const std::int64_t n = (std::int64_t)dim * hw;
    rms_channels(in, wts.mid_attn_norm, dim, h, w, ws.a, s);
    conv1x1(ws.a, wts.mid_attn_to_qkv, wts.mid_attn_to_qkv_b, dim, 3 * dim, hw, ws.e, s);
    const hip_bfloat16* q = ws.e;
    const hip_bfloat16* k = ws.e + n;
    const hip_bfloat16* v = ws.e + 2 * n;
    dim3 blk(256);
    attn_row_kernel<<<hw, blk, 0, s>>>(q, k, dim, hw, ws.f);
    dim3 grd((n + 255) / 256);
    attn_out_kernel<<<grd, blk, 0, s>>>(ws.f, v, dim, hw, ws.g);
    conv1x1(ws.g, wts.mid_attn_proj, wts.mid_attn_proj_b, dim, dim, hw, ws.a, s);
    add(ws.a, in, n, out, s);
}

// First-frame DupUp3D. Repeats channels, then unfolds them into a 2x spatial
// (and temporal) grid. The first temporal copy is dropped, which is what
// first_chunk=True does for a single image.
__global__ void dup_up_kernel(const hip_bfloat16* __restrict__ in, int cout, int h, int w,
                              int factor_t, int factor_s, int repeats, hip_bfloat16* __restrict__ out) {
    const int xx = blockIdx.x * blockDim.x + threadIdx.x;
    const int yy = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z;
    const int oh = h * factor_s;
    const int ow = w * factor_s;
    if (xx >= ow || yy >= oh || oc >= cout) return;
    const int hs = yy / factor_s;
    const int fs1 = yy - hs * factor_s;
    const int ws = xx / factor_s;
    const int fs2 = xx - ws * factor_s;
    const int ft = factor_t - 1;
    const int ch_rep = ((oc * factor_t + ft) * factor_s + fs1) * factor_s + fs2;
    const int src = ch_rep / repeats;
    out[(std::int64_t)oc * oh * ow + (std::int64_t)yy * ow + xx] =
        in[(std::int64_t)src * h * w + (std::int64_t)hs * w + ws];
}
void dup_up(const hip_bfloat16* in, int cin, int cout, int h, int w, int factor_t, int factor_s,
            hip_bfloat16* out, hipStream_t s) {
    const int factor = factor_t * factor_s * factor_s;
    const int repeats = cout * factor / cin;
    dim3 block(8, 8, 1);
    dim3 grid((w * factor_s + 7) / 8, (h * factor_s + 7) / 8, cout);
    dup_up_kernel<<<grid, block, 0, s>>>(in, cout, h, w, factor_t, factor_s, repeats, out);
}

} // namespace detail

namespace detail {

// Packed DiT latent [H*W, 64] -> NCHW [64, H, W], then z * std + mean.
// The tables live in the host function below. A file-scope array in this .hip
// file is a device symbol, and copying it with hipMemcpyHostToDevice uploads zeros.
__global__ void scale_latent_kernel(const hip_bfloat16* __restrict__ packed, int h, int w,
                                    const float* __restrict__ mean, const float* __restrict__ stdv,
                                    hip_bfloat16* __restrict__ nchw) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z;
    if (x >= w || y >= h || c >= 64) return;
    const float z = float(packed[((std::int64_t)y * w + x) * 64 + c]);
    const float v = z * stdv[c] + mean[c];
    nchw[(std::int64_t)c * h * w + (std::int64_t)y * w + x] = hip_bfloat16(v);
}

// Reference `_decode` clamps to [-1, 1]. A non-finite sample stays out of that
// range and used to crash the PNG writer; map it to 0.
__global__ void clamp_image_kernel(hip_bfloat16* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = float(x[i]);
    if (!(v == v)) v = 0.0f;
    else if (v < -1.0f) v = -1.0f;
    else if (v > 1.0f) v = 1.0f;
    x[i] = hip_bfloat16(v);
}

void vae_decode(const VaeWeights& w, const Tensor& latent, Tensor& out_image,
                VaeWorkspace& ws, hipStream_t s) {
    const int h0 = latent.ne[2], w0 = latent.ne[3];

    static float* d_mean = nullptr;
    static float* d_std = nullptr;
    if (d_mean == nullptr) {
        // Function-local, so hipcc keeps these in host memory.
        const float mean_host[64] = {0.51260000f, 0.77210000f, -0.06310000f, 1.35060000f, -0.78550000f, -2.10250000f, -0.34580000f, 1.37220000f, 1.88730000f, -1.71770000f, -0.65100000f, 0.27320000f, 0.75620000f, -0.61630000f, -1.02770000f, 3.83630000f, 2.02100000f, 0.04720000f, 0.93200000f, 2.00870000f, 2.49540000f, -0.13910000f, -1.42490000f, 1.84640000f, -0.52360000f, 1.28260000f, 3.70460000f, -1.30350000f, 2.72860000f, -1.45180000f, -1.90360000f, -1.99550000f, -0.03420000f, -1.02650000f, -0.76360000f, 3.05550000f, 0.07460000f, -3.07510000f, -0.10760000f, 1.73760000f, -1.09140000f, -1.94350000f, -0.27840000f, -1.36800000f, 0.48090000f, -0.44330000f, 0.37640000f, 0.57290000f, -2.05950000f, 1.09600000f, -1.32600000f, -2.02110000f, -5.01790000f, 0.52750000f, 4.01620000f, 1.85050000f, 0.30260000f, 1.93730000f, 1.49370000f, 0.26320000f, 0.55470000f, -1.71210000f, -0.15620000f, 0.03040000f};
        const float std_host[64] = {3.20010000f, 3.29360000f, 3.43210000f, 3.00910000f, 3.10610000f, 4.03790000f, 4.07050000f, 3.79100000f, 3.07850000f, 3.65000000f, 3.93080000f, 3.09040000f, 2.87780000f, 3.76750000f, 3.73200000f, 5.07560000f, 3.28640000f, 4.03970000f, 3.13170000f, 4.04430000f, 2.92490000f, 3.94540000f, 3.09880000f, 4.24890000f, 3.48960000f, 3.85130000f, 3.93230000f, 3.47190000f, 3.74980000f, 4.28300000f, 3.56940000f, 4.24670000f, 3.90370000f, 3.29470000f, 5.07700000f, 3.50750000f, 3.27000000f, 3.47670000f, 2.80630000f, 5.11250000f, 3.53270000f, 4.78330000f, 3.12860000f, 4.18190000f, 3.85270000f, 3.83120000f, 3.56050000f, 4.38750000f, 3.96240000f, 4.01680000f, 3.56430000f, 4.05500000f, 5.56140000f, 4.29630000f, 4.40800000f, 3.49590000f, 3.87470000f, 3.76080000f, 3.57350000f, 3.14900000f, 3.76620000f, 3.67460000f, 3.45630000f, 3.81610000f};
        float* mean = nullptr;
        float* stdv = nullptr;
        HIP_CHECK(hipMalloc(&mean, sizeof(mean_host)));
        HIP_CHECK(hipMalloc(&stdv, sizeof(std_host)));
        HIP_CHECK(hipMemcpyAsync(mean, mean_host, sizeof(mean_host), hipMemcpyHostToDevice, s));
        HIP_CHECK(hipMemcpyAsync(stdv, std_host, sizeof(std_host), hipMemcpyHostToDevice, s));
        float uploaded = 0.0f;
        HIP_CHECK(hipMemcpyAsync(&uploaded, stdv, sizeof(float), hipMemcpyDeviceToHost, s));
        HIP_CHECK(hipStreamSynchronize(s));
        // A zero table decodes every latent as the VAE's average color.
        if (!(uploaded > 1.0f)) {
            throw std::runtime_error("vae latent scale was not uploaded");
        }
        d_mean = mean;
        d_std = stdv;
    }
    scale_latent_kernel<<<dim3((w0 + 7) / 8, (h0 + 7) / 8, 64), dim3(8, 8, 1), 0, s>>>(
        reinterpret_cast<const hip_bfloat16*>(latent.data), h0, w0, d_mean, d_std, ws.b);
    conv1x1(ws.b, w.post_quant, w.post_quant_b, kLatentCh, kLatentCh, h0 * w0, ws.c, s);

    Tensor lat = t3(ws.c, kLatentCh, h0, w0),
           win = t4(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(w.conv_in)), kMidDim, kLatentCh, 3, 3),
           bin = t1(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(w.conv_in_b)), kMidDim),
           a0 = t3(ws.a, kMidDim, h0, w0);
    ops::conv2d_3x3(lat, win, bin, a0, s);

    // Mid block is resnet, attention, resnet. Attention adds its own residual.
    resnet(w.mid_resnets[0], ws.a, kMidDim, kMidDim, h0, w0, ws.b, ws, s);
    mid_attention(w, ws.b, kMidDim, h0, w0, ws.c, ws, s);
    resnet(w.mid_resnets[1], ws.c, kMidDim, kMidDim, h0, w0, ws.d, ws, s);

    hip_bfloat16* cur = ws.d; int ch = kMidDim; int hh = h0, ww = w0;
    for (int i = 0; i < 5; ++i) {
        const int cin = kUpIn[i];
        const int cout = kUpOut[i];
        // Blocks 0..3 add a duplicate-upsample of their input after the conv upsample.
        // factor_t is 2 on the 3d blocks (0..2) and 1 on the 2d block (3).
        if (i < 4) copy_bf16(cur, (std::int64_t)cin * hh * ww, ws.h, s);
        resnet(w.up_blocks[i].resnets[0], cur, cin, cout, hh, ww, ws.a, ws, s);
        resnet(w.up_blocks[i].resnets[1], ws.a, cout, cout, hh, ww, ws.b, ws, s);
        resnet(w.up_blocks[i].resnets[2], ws.b, cout, cout, hh, ww, ws.c, ws, s);
        cur = ws.c; ch = cout;
        if (i < 4) {
            const int factor_t = (i < 3) ? 2 : 1;
            upsample(w.up_blocks[i].upsampler, cur, ch, hh, ww, ws.e, ws, s);
            dup_up(ws.h, cin, cout, hh, ww, factor_t, 2, ws.f, s);
            add(ws.e, ws.f, (std::int64_t)cout * (2 * hh) * (2 * ww), ws.e, s);
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
    const int nimg = kImageCh * hh * ww;
    clamp_image_kernel<<<(nimg + 255) / 256, 256, 0, s>>>(
        reinterpret_cast<hip_bfloat16*>(out_image.data), nimg);
}


// =========================================================================
// 2D Encoder (for img2img): image [4, H, W] in [-1,1] -> latent [64, H/16, W/16]
// =========================================================================

// Downsampler: zero-pad (right=1, bottom=1) + 3x3 conv stride 2.
// in: [C, H, W] -> out: [C, H/2, W/2].
__global__ void downsample2d_kernel(const hip_bfloat16* __restrict__ in, const hip_bfloat16* __restrict__ w,
                                    const hip_bfloat16* __restrict__ b, int c, int h, int w_in,
                                    hip_bfloat16* __restrict__ out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int co = blockIdx.z;
    const int w_out = w_in / 2, h_out = h / 2;
    if (x >= w_out || y >= h_out || co >= c) return;
    float acc = float(b[co]);
    for (int ci = 0; ci < c; ++ci) {
        for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
                const int iy = y * 2 + ky, ix = x * 2 + kx;
                if (iy >= h || ix >= w_in) continue;  // zero-pad on right/bottom
                acc += float(w[(std::int64_t)co * c * 9 + (std::int64_t)ci * 9 + ky * 3 + kx]) *
                       float(in[(std::int64_t)ci * h * w_in + (std::int64_t)iy * w_in + ix]);
            }
        }
    }
    out[(std::int64_t)co * h_out * w_out + (std::int64_t)y * w_out + x] = hip_bfloat16(acc);
}
void downsample2d(const hip_bfloat16* in, const hip_bfloat16* w, const hip_bfloat16* b,
                  int c, int h, int w_in, hip_bfloat16* out, hipStream_t s) {
    dim3 blk(8, 8, 1);
    dim3 grd((w_in / 2 + 7) / 8, (h / 2 + 7) / 8, c);
    downsample2d_kernel<<<grd, blk, 0, s>>>(in, w, b, c, h, w_in, out);
}

// AvgDown shortcut: 2x spatial avg-pool + channel grouping.
// in: [C, H, W] -> out: [cout, H/2, W/2]. group = C*4 / cout.
// out[oc,y,x] = (1/group) * sum_g in_pooled[oc*group+g, y, x]
//   where in_pooled[ch,y,x] = in[ch/4, 2y+(ch%4)/2, 2x+(ch%4)%2]
__global__ void avg_down_kernel(const hip_bfloat16* __restrict__ in, int h, int w, int cout, int group,
                                hip_bfloat16* __restrict__ out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z;
    const int w_out = w / 2, h_out = h / 2;
    if (x >= w_out || y >= h_out || oc >= cout) return;
    float acc = 0.0f;
    for (int g = 0; g < group; ++g) {
        const int ch = oc * group + g;      // index in [C*4] space
        const int ci = ch / 4;
        const int fy = (ch % 4) / 2;
        const int fx = (ch % 4) % 2;
        acc += float(in[(std::int64_t)ci * h * w + (std::int64_t)(y * 2 + fy) * w + (x * 2 + fx)]);
    }
    out[(std::int64_t)oc * h_out * w_out + (std::int64_t)y * w_out + x] = hip_bfloat16(acc / (float)group);
}
void avg_down(const hip_bfloat16* in, int c, int h, int w, int cout, hip_bfloat16* out, hipStream_t s) {
    const int group = c * 4 / cout;
    dim3 blk(8, 8, 1);
    dim3 grd((w / 2 + 7) / 8, (h / 2 + 7) / 8, cout);
    avg_down_kernel<<<grd, blk, 0, s>>>(in, h, w, cout, group, out);
}


// [debug] dump mean/std of a [c,h,w] bf16 map to /tmp/enc_act.log

// Identity channel-group (block 4: factor_s=1, group=1). Just a copy.
void vae_encode(const VaeWeights& w, const Tensor& image, Tensor& out_latent,
                VaeWorkspace& ws, hipStream_t s) {
    const int H = image.ne[2], W = image.ne[3];
    const int C0 = 4;
    // dims: [96, 96, 192, 384, 768, 768]; blocks 0-3 downsample, block 4 does not.
    const int dims[6] = {96, 96, 192, 384, 768, 768};

    // conv_in: 4 -> 96
    {
        Tensor in_t = t3(reinterpret_cast<const hip_bfloat16*>(image.data), C0, H, W);
        Tensor win = t4(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(w.enc_conv_in)), 96, C0, 3, 3);
        Tensor bin = t1(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(w.enc_conv_in_b)), 96);
        Tensor a0 = t3(ws.a, 96, H, W);
        ops::conv2d_3x3(in_t, win, bin, a0, s);
    }
    hip_bfloat16* cur = ws.a; int ch = 96; int hh = H, ww = W;
    for (int i = 0; i < 5; ++i) {
        const int cin = dims[i], cout = dims[i + 1];
        // avg_shortcut input (clone of cur before resnets)
        copy_bf16(cur, (std::int64_t)cin * hh * ww, ws.h, s);
        resnet(w.enc_down_blocks[i].resnets[0], cur, cin, cout, hh, ww, ws.a, ws, s);
        resnet(w.enc_down_blocks[i].resnets[1], ws.a, cout, cout, hh, ww, ws.b, ws, s);
        cur = ws.b;
        if (w.enc_down_blocks[i].has_down) {
            downsample2d(cur, w.enc_down_blocks[i].resample, w.enc_down_blocks[i].resample_b,
                         cout, hh, ww, ws.c, s);
            cur = ws.c; hh /= 2; ww /= 2;
        }
        // add avg_shortcut(ws.h)
        if (i < 4) {
            avg_down(ws.h, cin, hh * 2, ww * 2, cout, ws.d, s);
            add(cur, ws.d, (std::int64_t)cout * hh * ww, cur, s);
        } else {
            // block 4: identity shortcut (group=1, no spatial change)
            add(cur, ws.h, (std::int64_t)cout * hh * ww, cur, s);
        }
            }
    ch = dims[5];  // 768: output of the last down_block
    // mid_block: resnet, attention, resnet
    resnet(w.enc_mid_resnets[0], cur, ch, ch, hh, ww, ws.a, ws, s);
    // Encoder-specific attention (reads w.enc_mid_attn_*). Input ws.a, output ws.b.
    {
        const int dim = ch, hw = hh * ww;
        const std::int64_t n = (std::int64_t)dim * hw;
        copy_bf16(ws.a, n, ws.c, s);  // preserve input for the residual
        rms_channels(ws.c, w.enc_mid_attn_norm, dim, hh, ww, ws.a, s);
        conv1x1(ws.a, w.enc_mid_attn_to_qkv, w.enc_mid_attn_to_qkv_b, dim, 3 * dim, hw, ws.e, s);
        const hip_bfloat16* q = ws.e;
        const hip_bfloat16* k = ws.e + n;
        const hip_bfloat16* v = ws.e + 2 * n;
        dim3 blk(256);
        attn_row_kernel<<<hw, blk, 0, s>>>(q, k, dim, hw, ws.f);
        dim3 grd((n + 255) / 256);
        attn_out_kernel<<<grd, blk, 0, s>>>(ws.f, v, dim, hw, ws.g);
        conv1x1(ws.g, w.enc_mid_attn_proj, w.enc_mid_attn_proj_b, dim, dim, hw, ws.a, s);
        add(ws.a, ws.c, n, ws.b, s);  // attention out + input residual
    }
    resnet(w.enc_mid_resnets[1], ws.b, ch, ch, hh, ww, ws.c, ws, s);
    cur = ws.c;
    // norm_out + silu + conv_out (768 -> 128)
    rms_channels(cur, w.enc_norm_out, ch, hh, ww, ws.c, s);
    silu(ws.c, (std::int64_t)ch * hh * ww, ws.d, s);
    {
        Tensor in_t = t3(ws.d, ch, hh, ww);
        Tensor win = t4(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(w.enc_conv_out)), 128, ch, 3, 3);
        Tensor bin = t1(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(w.enc_conv_out_b)), 128);
        Tensor a0 = t3(ws.e, 128, hh, ww);
        ops::conv2d_3x3(in_t, win, bin, a0, s);
    }
    // quant_conv (1x1, 128 -> 128)
    conv1x1(ws.e, w.enc_quant_conv, w.enc_quant_conv_b, 128, 128, hh * ww, ws.f, s);
    // mode: take first 64 channels -> out_latent [64, hh, ww]
    {
        const int out_h = hh, out_w = ww;
        // ws.f is [128, hh, ww] NCHW. Take channels 0..63 (the first 64*hw*2 bytes).
        (void)hipMemcpyAsync(out_latent.data, ws.f, (std::int64_t)64 * out_h * out_w * 2,
                             hipMemcpyDeviceToDevice, s);
    }
}

} // namespace detail
} // namespace ninfer::targets::qwen_image_vae
