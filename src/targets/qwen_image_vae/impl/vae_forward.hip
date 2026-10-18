// VAE decode forward: latent [1, 64, H/16, W/16] -> image [1, 4, H, W].
// The decoder: conv_in -> mid_block -> up_blocks (5) -> norm_out -> SiLU -> conv_out.
// NCHW layout (single image). The conv2d_3x3 op handles NCHW directly.
#include "vae_forward.h"

#include "ninfer/ops/conv2d_3x3.h"
#include "ninfer/ops/upsample2d_nearest.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace ninfer::targets::qwen_image_vae {
namespace {

constexpr int kLatentCh = 64;
[[maybe_unused]] constexpr int kImageCh = 4; // used in conv_out (TODO)
constexpr int kMidDim   = 1152;
constexpr int kUpOut[5] = {1152, 1152, 576, 288, 144};
constexpr int kUpIn[5]  = {1152, 1152, 1152, 576, 288};

// Helper: make a Tensor from a raw pointer + shape (NCHW)
inline Tensor make_tensor(void* data, int c, int h, int w) {
    Tensor t;
    t.data = data;
    t.dtype = DType::BF16;
    t.ne[0] = c;
    t.ne[1] = h;
    t.ne[2] = w;
    t.ne[3] = 1;
    t.nb[0] = 2;
    t.nb[1] = (std::int64_t)c * w * 2;
    t.nb[2] = (std::int64_t)c * h * w * 2;
    t.nb[3] = 0;
    return t;
}

inline Tensor make_tensor_4d(void* data, int d0, int d1, int d2, int d3) {
    Tensor t;
    t.data = data;
    t.dtype = DType::BF16;
    t.ne[0] = d0;
    t.ne[1] = d1;
    t.ne[2] = d2;
    t.ne[3] = d3;
    t.nb[0] = 2;
    t.nb[1] = (std::int64_t)d0 * 2;
    t.nb[2] = (std::int64_t)d0 * d1 * 2;
    t.nb[3] = (std::int64_t)d0 * d1 * d2 * 2;
    return t;
}

inline Tensor make_tensor_1d(void* data, int n) {
    Tensor t;
    t.data = data;
    t.dtype = DType::BF16;
    t.ne[0] = n;
    t.ne[1] = 1;
    t.ne[2] = 1;
    t.ne[3] = 1;
    t.nb[0] = 2;
    t.nb[1] = (std::int64_t)n * 2;
    t.nb[2] = (std::int64_t)n * 2;
    t.nb[3] = (std::int64_t)n * 2;
    return t;
}

// SiLU activation (element-wise): out = x * sigmoid(x)
__global__ void silu_kernel(const hip_bfloat16* __restrict__ in, std::int64_t n,
                            hip_bfloat16* __restrict__ out) {
    std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = float(in[i]);
    out[i] = hip_bfloat16(x / (1.0f + expf(-x)));
}

void silu(const hip_bfloat16* in, std::int64_t n, hip_bfloat16* out, hipStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    silu_kernel<<<grid, block, 0, stream>>>(in, n, out);
}

// Element-wise add: out = a + b (same shape)
__global__ void add_kernel(const hip_bfloat16* __restrict__ a, const hip_bfloat16* __restrict__ b,
                           std::int64_t n, hip_bfloat16* __restrict__ out) {
    std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = hip_bfloat16(float(a[i]) + float(b[i]));
}

void add(const hip_bfloat16* a, const hip_bfloat16* b, std::int64_t n, hip_bfloat16* out,
         hipStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    add_kernel<<<grid, block, 0, stream>>>(a, b, n, out);
}

// LayerNorm over channels (C, H, W): normalize over C for each (h, w) position.
__global__ void layernorm_channels_kernel(const hip_bfloat16* __restrict__ in,
                                          const hip_bfloat16* __restrict__ weight,
                                          int c, int h, int w, float eps,
                                          hip_bfloat16* __restrict__ out) {
    const int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= h * w) return;
    const int y = hw / w;
    const int x = hw % w;

    float sum = 0.0f, sum_sq = 0.0f;
    for (int ci = 0; ci < c; ++ci) {
        float v = float(in[(std::int64_t)ci * h * w + (std::int64_t)y * w + x]);
        sum += v;
        sum_sq += v * v;
    }
    const float mean = sum / c;
    const float var = sum_sq / c - mean * mean;
    const float inv = 1.0f / sqrtf(var + eps);

    for (int ci = 0; ci < c; ++ci) {
        float v = float(in[(std::int64_t)ci * h * w + (std::int64_t)y * w + x]);
        float wv = float(weight[ci]);
        out[(std::int64_t)ci * h * w + (std::int64_t)y * w + x] =
            hip_bfloat16((v - mean) * inv * wv);
    }
}

void layernorm_channels(const hip_bfloat16* in, const hip_bfloat16* weight, int c, int h, int w,
                        float eps, hip_bfloat16* out, hipStream_t stream) {
    dim3 block(256);
    dim3 grid((h * w + 255) / 256);
    layernorm_channels_kernel<<<grid, block, 0, stream>>>(in, weight, c, h, w, eps, out);
}

// Residual block: norm1 -> SiLU -> conv1 -> norm2 -> SiLU -> conv2 -> + residual
void resnet(const VaeWeights::Resnet& r, const hip_bfloat16* in, int cin, int cout, int h, int w,
            hip_bfloat16* out, VaeWorkspace& ws, hipStream_t stream) {
    const std::int64_t n_in  = (std::int64_t)cin * h * w;
    const std::int64_t n_out = (std::int64_t)cout * h * w;

    // norm1 (LayerNorm over channels)
    layernorm_channels(in, r.norm1, cin, h, w, 1e-5f, ws.a, stream);
    // SiLU
    silu(ws.a, n_in, ws.b, stream);
    // conv1
    Tensor b_tensor = make_tensor(ws.b, cin, h, w);
    Tensor c_tensor = make_tensor(ws.c, cout, h, w);
    Tensor w1 = make_tensor_4d(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(r.conv1)), cout, cin, 3, 3);
    Tensor b1 = make_tensor_1d(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(r.conv1_b)), cout);
    ops::conv2d_3x3(b_tensor, w1, b1, c_tensor, stream);
    // norm2
    layernorm_channels(ws.c, r.norm2, cout, h, w, 1e-5f, ws.d, stream);
    // SiLU
    silu(ws.d, n_out, ws.e, stream);
    // conv2
    Tensor e_tensor = make_tensor(ws.e, cout, h, w);
    Tensor f_tensor = make_tensor(ws.f, cout, h, w);
    Tensor w2 = make_tensor_4d(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(r.conv2)), cout, cout, 3, 3);
    Tensor b2 = make_tensor_1d(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(r.conv2_b)), cout);
    ops::conv2d_3x3(e_tensor, w2, b2, f_tensor, stream);
    // residual: if shortcut exists, apply it to in; else add in directly
    // (For now, just add in directly - the shortcut is a TODO)
    add(ws.f, in, n_out, out, stream);
}

} // namespace

void vae_decode(const VaeWeights& weights, const Tensor& latent, Tensor& out_image,
                VaeWorkspace& ws, hipStream_t stream) {
    // latent: [1, 64, H/16, W/16] (NCHW, single image)
    const int h0 = latent.ne[2];
    const int w0 = latent.ne[3];

    // conv_in: [64, H/16, W/16] -> [1152, H/16, W/16]
    Tensor latent_view = make_tensor(latent.data, kLatentCh, h0, w0);
    Tensor a_tensor = make_tensor(ws.a, kMidDim, h0, w0);
    Tensor w_in = make_tensor_4d(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(weights.conv_in)), kMidDim, kLatentCh, 3, 3);
    Tensor b_in = make_tensor_1d(reinterpret_cast<void*>(const_cast<hip_bfloat16*>(weights.conv_in_b)), kMidDim);
    ops::conv2d_3x3(latent_view, w_in, b_in, a_tensor, stream);

    // mid_block: 2 resnets (simplified: no attention for now)
    resnet(weights.mid_resnets[0], ws.a, kMidDim, kMidDim, h0, w0, ws.b, ws, stream);
    resnet(weights.mid_resnets[1], ws.b, kMidDim, kMidDim, h0, w0, ws.c, ws, stream);

    // up_blocks (5 blocks) - simplified: just the first up_block for now
    resnet(weights.up_blocks[0].resnets[0], ws.c, kUpIn[0], kUpOut[0], h0, w0, ws.d, ws, stream);
    resnet(weights.up_blocks[0].resnets[1], ws.d, kUpOut[0], kUpOut[0], h0, w0, ws.e, ws, stream);
    resnet(weights.up_blocks[0].resnets[2], ws.e, kUpOut[0], kUpOut[0], h0, w0, ws.f, ws, stream);

    // conv_out: [144, H, W] -> [4, H, W] (simplified: just copy for now)
    // For now, just copy ws.f to out_image (the full decode is a TODO)
    const std::int64_t n = (std::int64_t)kUpOut[0] * h0 * w0;
    (void)hipMemcpyAsync(out_image.data, ws.f, (std::size_t)n * 2, hipMemcpyDeviceToDevice);
}

} // namespace ninfer::targets::qwen_image_vae
