// Target-local HIP kernels for the Qwen-Image-2.1 DiT (wave32 gfx1201).
#include "dit_ops.h"

#include <hip/hip_bfloat16.h>

#include <cstdint>

namespace ninfer::targets::qwen_image_21::detail {
namespace {

// 3D complex RoPE: 64 complex pairs per (token, head); 32 lanes -> 2 pairs/lane.
__global__ void dit_rope_kernel(const float* __restrict__ freqs, hip_bfloat16* __restrict__ q,
                                hip_bfloat16* __restrict__ k, std::int64_t q_batch,
                                std::int64_t q_stride, std::int64_t k_batch, std::int64_t k_stride) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int batch = blockIdx.z;
    const std::int64_t qbase =
        (std::int64_t)batch * q_batch + (std::int64_t)token * q_stride + (std::int64_t)head * 128;
    const std::int64_t kbase =
        (std::int64_t)batch * k_batch + (std::int64_t)token * k_stride + (std::int64_t)head * 128;
    const float* f = freqs + (std::int64_t)token * 128;
    for (int pair = threadIdx.x; pair < 64; pair += 32) {
        const float fre = f[2 * pair];
        const float fim = f[2 * pair + 1];
        {
            const float re = q[qbase + 2 * pair];
            const float im = q[qbase + 2 * pair + 1];
            q[qbase + 2 * pair]       = hip_bfloat16(re * fre - im * fim);
            q[qbase + 2 * pair + 1]   = hip_bfloat16(re * fim + im * fre);
        }
        {
            const float re = k[kbase + 2 * pair];
            const float im = k[kbase + 2 * pair + 1];
            k[kbase + 2 * pair]       = hip_bfloat16(re * fre - im * fim);
            k[kbase + 2 * pair + 1]   = hip_bfloat16(re * fim + im * fre);
        }
    }
}

// No-affine LayerNorm over the last dim (hidden). One block of 256 per (batch, token).
template <int Hidden>
__global__ void dit_layernorm_kernel(const hip_bfloat16* __restrict__ x, hip_bfloat16* __restrict__ out,
                                     float eps) {
    const int row = blockIdx.x;
    const hip_bfloat16* xr = x + (std::int64_t)row * Hidden;
    hip_bfloat16* orow     = out + (std::int64_t)row * Hidden;
    __shared__ float red[2][256 / 32];
    float sum = 0.0F, sumsq = 0.0F;
    float vals[Hidden / 256];
    for (int i = threadIdx.x; i < Hidden; i += 256) {
        const float v = xr[i];
        vals[i / 256] = v;
        sum   += v;
        sumsq += v * v;
    }
    // block reduce sum and sumsq
    for (int off = 16; off > 0; off >>= 1) {
        sum   += __shfl_down_sync(0xffffffffu, sum, off);
        sumsq += __shfl_down_sync(0xffffffffu, sumsq, off);
    }
    if ((threadIdx.x & 31) == 0) {
        red[0][threadIdx.x / 32] = sum;
        red[1][threadIdx.x / 32] = sumsq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0F, q = 0.0F;
        for (int w = 0; w < 256 / 32; w++) {
            s += red[0][w];
            q += red[1][w];
        }
        red[0][0] = s;
        red[1][0] = q;
    }
    __syncthreads();
    const float mean  = red[0][0] / Hidden;
    const float var   = red[1][0] / Hidden - mean * mean;
    const float rstd  = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < Hidden; i += 256) {
        orow[i] = hip_bfloat16((vals[i / 256] - mean) * rstd);
    }
}

// out = x * (1 + scale); scale [batch, hidden] broadcast over tokens.
__global__ void dit_modulate_kernel(const hip_bfloat16* __restrict__ x,
                                    const hip_bfloat16* __restrict__ scale,
                                    hip_bfloat16* __restrict__ out, int tokens, int hidden,
                                    int total) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= total) return;
    const int col   = idx % hidden;
    const int batch = (idx / hidden) / tokens;
    out[idx] = hip_bfloat16(x[idx] * (1.0F + scale[(std::int64_t)batch * hidden + col]));
}

// out = x + tanh(gate) * delta; gate [batch, hidden] broadcast.
__global__ void dit_residual_gate_kernel(const hip_bfloat16* __restrict__ x,
                                         const hip_bfloat16* __restrict__ gate,
                                         const hip_bfloat16* __restrict__ delta, hip_bfloat16* __restrict__ out,
                                         int tokens, int hidden, int total) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= total) return;
    const int col   = idx % hidden;
    const int batch = (idx / hidden) / tokens;
    const float g = tanhf(gate[(std::int64_t)batch * hidden + col]);
    out[idx] = hip_bfloat16(x[idx] + g * delta[idx]);
}

} // namespace

void dit_rope(const float* freqs, Tensor& q, Tensor& k, hipStream_t stream) {
    const int tokens = q.ne[1];
    const int heads  = q.ne[2];
    const int batch  = q.ne[0];
    const std::int64_t q_batch  = q.nb[0];
    const std::int64_t q_stride = q.nb[1];
    const std::int64_t k_batch  = k.nb[0];
    const std::int64_t k_stride = k.nb[1];
    dim3 grid(tokens, heads, batch);
    dim3 block(32);
    dit_rope_kernel<<<grid, block, 0, stream>>>(freqs, (hip_bfloat16*)q.data, (hip_bfloat16*)k.data,
                                                q_batch, q_stride, k_batch, k_stride);
}

void dit_layer_norm(const Tensor& x, float eps, Tensor& out, hipStream_t stream) {
    const int rows = x.ne[0] * x.ne[1];
    const int hidden = x.ne[2];
    dim3 grid(rows);
    dim3 block(256);
    if (hidden == 4096) {
        dit_layernorm_kernel<4096><<<grid, block, 0, stream>>>((const hip_bfloat16*)x.data,
                                                               (hip_bfloat16*)out.data, eps);
    } else {
        throw std::runtime_error("dit_layer_norm: unsupported hidden");
    }
}

void dit_modulate_scale(const Tensor& x, const Tensor& scale, Tensor& out, hipStream_t stream) {
    const int tokens = x.ne[1];
    const int hidden = x.ne[2];
    const int total  = x.ne[0] * tokens * hidden;
    dim3 grid((total + 255) / 256);
    dim3 block(256);
    dit_modulate_kernel<<<grid, block, 0, stream>>>((const hip_bfloat16*)x.data,
                                                    (const hip_bfloat16*)scale.data,
                                                    (hip_bfloat16*)out.data, tokens, hidden, total);
}

void dit_residual_gate(const Tensor& x, const Tensor& gate, const Tensor& delta, Tensor& out,
                       hipStream_t stream) {
    const int tokens = x.ne[1];
    const int hidden = x.ne[2];
    const int total  = x.ne[0] * tokens * hidden;
    dim3 grid((total + 255) / 256);
    dim3 block(256);
    dit_residual_gate_kernel<<<grid, block, 0, stream>>>((const hip_bfloat16*)x.data,
                                                         (const hip_bfloat16*)gate.data,
                                                         (const hip_bfloat16*)delta.data,
                                                         (hip_bfloat16*)out.data, tokens, hidden, total);
}

} // namespace ninfer::targets::qwen_image_21::detail
