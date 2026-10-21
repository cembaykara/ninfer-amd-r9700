// Target-local HIP kernels for the Qwen-Image-2.1 DiT (wave32 gfx1201).
#include "dit_ops.h"

#include "core/device.h"

#include <hip/hip_bfloat16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen_image_21::detail {
namespace {

// The rounding mirrors the PyTorch reference, which does each elementwise step in BF16.
__device__ __forceinline__ float bf(float v) { return static_cast<float>(hip_bfloat16(v)); }

__global__ void dit_rope_kernel(const float* __restrict__ freqs, hip_bfloat16* __restrict__ q,
                                hip_bfloat16* __restrict__ k, int heads) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int pair  = threadIdx.x;  // 64 complex pairs per head
    const std::int64_t base = ((std::int64_t)token * heads + head) * 128 + 2 * pair;
    const float c = freqs[(std::int64_t)token * 128 + 2 * pair];
    const float s = freqs[(std::int64_t)token * 128 + 2 * pair + 1];
    hip_bfloat16* const qk[2] = {q, k};
    for (hip_bfloat16* x : qk) {
        const float re = x[base];
        const float im = x[base + 1];
        x[base]     = hip_bfloat16(re * c - im * s);
        x[base + 1] = hip_bfloat16(re * s + im * c);
    }
}

__device__ __forceinline__ float block_sum_256(float v, float* red) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = v;
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < 8; w++) total += red[w];
    __syncthreads();
    return total;
}

template <int Hidden>
__global__ void dit_layernorm_kernel(const hip_bfloat16* __restrict__ x, hip_bfloat16* __restrict__ out,
                                     float eps) {
    const std::int64_t base = (std::int64_t)blockIdx.x * Hidden;
    __shared__ float red[8];
    float vals[Hidden / 256];
    float sum = 0.0F;
    for (int i = 0; i < Hidden / 256; i++) {
        vals[i] = x[base + threadIdx.x + i * 256];
        sum += vals[i];
    }
    const float mean = block_sum_256(sum, red) / Hidden;
    float sq = 0.0F;
    for (int i = 0; i < Hidden / 256; i++) {
        const float d = vals[i] - mean;
        sq += d * d;
    }
    const float rstd = rsqrtf(block_sum_256(sq, red) / Hidden + eps);
    for (int i = 0; i < Hidden / 256; i++) {
        out[base + threadIdx.x + i * 256] = hip_bfloat16((vals[i] - mean) * rstd);
    }
}

__global__ void dit_modulate_kernel(hip_bfloat16* __restrict__ x, const hip_bfloat16* __restrict__ img,
                                    const hip_bfloat16* __restrict__ txt, int M, int hidden,
                                    std::int64_t total) {
    const std::int64_t idx = (std::int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= total) return;
    const int col = (int)(idx % hidden);
    const hip_bfloat16* scale = (idx / hidden) < M ? txt : img;
    x[idx] = hip_bfloat16((float)x[idx] * bf(1.0F + (float)scale[col]));
}

__global__ void dit_residual_gate_kernel(hip_bfloat16* __restrict__ x, const hip_bfloat16* __restrict__ img,
                                         const hip_bfloat16* __restrict__ txt, int M,
                                         const hip_bfloat16* __restrict__ delta, int hidden,
                                         std::int64_t total) {
    const std::int64_t idx = (std::int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= total) return;
    const int col = (int)(idx % hidden);
    const hip_bfloat16* gate = (idx / hidden) < M ? txt : img;
    const float g = bf(tanhf((float)gate[col]));
    x[idx] = hip_bfloat16((float)x[idx] + bf(g * (float)delta[idx]));
}

__global__ void dit_silu_kernel(const hip_bfloat16* __restrict__ x, hip_bfloat16* __restrict__ out,
                                std::int64_t total) {
    const std::int64_t idx = (std::int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= total) return;
    const float v = x[idx];
    out[idx] = hip_bfloat16(v / (1.0F + expf(-v)));
}

__global__ void dit_euler_kernel(hip_bfloat16* __restrict__ latent, const hip_bfloat16* __restrict__ velocity,
                                 float dt, std::int64_t total) {
    const std::int64_t idx = (std::int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= total) return;
    latent[idx] = hip_bfloat16((float)latent[idx] + dt * (float)velocity[idx]);
}

unsigned int blocks_for(std::int64_t total) { return (unsigned int)((total + 255) / 256); }

} // namespace

void dit_rope(const float* freqs, Tensor& q, Tensor& k, hipStream_t stream) {
    if (q.ne[0] != 128 || k.ne[0] != 128) throw std::runtime_error("dit_rope: head_dim must be 128");
    dim3 grid(q.ne[2], q.ne[1]);
    dit_rope_kernel<<<grid, 64, 0, stream>>>(freqs, (hip_bfloat16*)q.data, (hip_bfloat16*)k.data,
                                             (int)q.ne[1]);
}

void dit_layer_norm(const Tensor& x, float eps, Tensor& out, hipStream_t stream) {
    if (x.ne[0] != 4096) throw std::runtime_error("dit_layer_norm: unsupported hidden");
    dit_layernorm_kernel<4096><<<x.ne[1], 256, 0, stream>>>((const hip_bfloat16*)x.data,
                                                            (hip_bfloat16*)out.data, eps);
}

void dit_modulate(Tensor& x, const hip_bfloat16* img, const hip_bfloat16* txt, int M,
                  hipStream_t stream) {
    const std::int64_t total = x.numel();
    dit_modulate_kernel<<<blocks_for(total), 256, 0, stream>>>((hip_bfloat16*)x.data, img, txt, M,
                                                               (int)x.ne[0], total);
}

void dit_residual_gate(Tensor& x, const hip_bfloat16* img, const hip_bfloat16* txt, int M,
                       const Tensor& delta, hipStream_t stream) {
    const std::int64_t total = x.numel();
    dit_residual_gate_kernel<<<blocks_for(total), 256, 0, stream>>>(
        (hip_bfloat16*)x.data, img, txt, M, (const hip_bfloat16*)delta.data, (int)x.ne[0], total);
}

void dit_silu(const Tensor& x, Tensor& out, hipStream_t stream) {
    const std::int64_t total = x.numel();
    dit_silu_kernel<<<blocks_for(total), 256, 0, stream>>>((const hip_bfloat16*)x.data,
                                                           (hip_bfloat16*)out.data, total);
}

void dit_euler(Tensor& latent, const Tensor& velocity, float dt, hipStream_t stream) {
    const std::int64_t total = latent.numel();
    dit_euler_kernel<<<blocks_for(total), 256, 0, stream>>>(
        (hip_bfloat16*)latent.data, (const hip_bfloat16*)velocity.data, dt, total);
}

// One workgroup owns 128 rows by 512 tokens. The 32 waves cover a 64x128 base and
// each wave holds the other four token strips, plus a second row strip, in distinct
// registers. The weight slice is loaded once per K step and reused across all 512
// tokens. An array of accumulators spills and miscompiles the WMMA operand, so each
// accumulator is a named fragment.
constexpr int kBaseM = 64;
constexpr int kBaseT = 128;
constexpr int kGemmM = 128;
constexpr int kGemmT = 512;
constexpr int kGemmK = 64;
// Pad each shared-memory row. A stride of 64 bf16 puts every lane on the same banks.
constexpr int kLdsK  = kGemmK + 8;
constexpr int kGemmThreads = 1024;

using GemmBf16 = short __attribute__((ext_vector_type(8)));
using GemmF32  = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void dit_gemm_load_x(const hip_bfloat16* x, hip_bfloat16* tile, int t0,
                                                int k0, int T, int K, int tid) {
    constexpr int x_vecs = kBaseT * kGemmK / 8;
    for (int vec = tid; vec < x_vecs; vec += kGemmThreads) {
        const int elem = vec * 8;
        const int lt   = elem / kGemmK;
        const int lk   = elem - lt * kGemmK;
        const int gt   = t0 + lt;
        uint4 bits{};
        if (gt < T) {
            bits = *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(gt) * K + k0 + lk);
        }
        *reinterpret_cast<uint4*>(&tile[lt * kLdsK + lk]) = bits;
    }
}

__device__ __forceinline__ GemmF32 dit_gemm_mma(const hip_bfloat16* tile_x, const hip_bfloat16* tile_w,
                                                GemmF32 acc, int an, int wave_t, int wave_n, int axis,
                                                int group) {
    for (int kk = 0; kk < kGemmK; kk += 16) {
        const int local_t = wave_t * 16 + axis;
        const int local_n = an * kBaseM + wave_n * 16 + axis;
        const GemmBf16 a =
            *reinterpret_cast<const GemmBf16*>(&tile_x[local_t * kLdsK + kk + group * 8]);
        const GemmBf16 b =
            *reinterpret_cast<const GemmBf16*>(&tile_w[local_n * kLdsK + kk + group * 8]);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc);
    }
    return acc;
}

__device__ __forceinline__ void dit_gemm_store(const GemmF32& acc, hip_bfloat16* out, int t0, int row,
                                               int T, int N, int wave_t, int group) {
    if (row >= N) return;
#pragma unroll
    for (int element = 0; element < 8; ++element) {
        const int token = t0 + wave_t * 16 + group * 8 + element;
        if (token < T) out[static_cast<std::int64_t>(token) * N + row] = hip_bfloat16(acc[element]);
    }
}

__global__ __launch_bounds__(kGemmThreads, 1) void dit_gemm_kernel(const hip_bfloat16* __restrict__ x,
                                                                   const hip_bfloat16* __restrict__ w,
                                                                   hip_bfloat16* __restrict__ out, int T,
                                                                   int N, int K) {
    __shared__ __attribute__((aligned(16))) hip_bfloat16 tile_x[kBaseT * kLdsK];
    __shared__ __attribute__((aligned(16))) hip_bfloat16 tile_w[kGemmM * kLdsK];

    const int n0     = static_cast<int>(blockIdx.x) * kGemmM;
    const int t0     = static_cast<int>(blockIdx.y) * kGemmT;
    const int tid    = static_cast<int>(threadIdx.x);
    const int wave   = tid / 32;
    const int lane   = tid % 32;
    const int wave_n = wave % (kBaseM / 16);
    const int wave_t = wave / (kBaseM / 16);
    const int axis   = lane % 16;
    const int group  = lane / 16;
    const int row0   = n0 + wave_n * 16 + axis;
    const int row1   = row0 + kBaseM;

    GemmF32 c00{}, c01{}, c02{}, c03{};
    GemmF32 c10{}, c11{}, c12{}, c13{};

    constexpr int w_vecs = kGemmM * kGemmK / 8;
    for (int k0 = 0; k0 < K; k0 += kGemmK) {
        for (int vec = tid; vec < w_vecs; vec += kGemmThreads) {
            const int elem = vec * 8;
            const int ln   = elem / kGemmK;
            const int lk   = elem - ln * kGemmK;
            const int gn   = n0 + ln;
            uint4 bits{};
            if (gn < N) {
                bits = *reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(gn) * K + k0 + lk);
            }
            *reinterpret_cast<uint4*>(&tile_w[ln * kLdsK + lk]) = bits;
        }
        __syncthreads();

        dit_gemm_load_x(x, tile_x, t0 + 0 * kBaseT, k0, T, K, tid);
        __syncthreads();
        c00 = dit_gemm_mma(tile_x, tile_w, c00, 0, wave_t, wave_n, axis, group);
        c10 = dit_gemm_mma(tile_x, tile_w, c10, 1, wave_t, wave_n, axis, group);
        __syncthreads();
        dit_gemm_load_x(x, tile_x, t0 + 1 * kBaseT, k0, T, K, tid);
        __syncthreads();
        c01 = dit_gemm_mma(tile_x, tile_w, c01, 0, wave_t, wave_n, axis, group);
        c11 = dit_gemm_mma(tile_x, tile_w, c11, 1, wave_t, wave_n, axis, group);
        __syncthreads();
        dit_gemm_load_x(x, tile_x, t0 + 2 * kBaseT, k0, T, K, tid);
        __syncthreads();
        c02 = dit_gemm_mma(tile_x, tile_w, c02, 0, wave_t, wave_n, axis, group);
        c12 = dit_gemm_mma(tile_x, tile_w, c12, 1, wave_t, wave_n, axis, group);
        __syncthreads();
        dit_gemm_load_x(x, tile_x, t0 + 3 * kBaseT, k0, T, K, tid);
        __syncthreads();
        c03 = dit_gemm_mma(tile_x, tile_w, c03, 0, wave_t, wave_n, axis, group);
        c13 = dit_gemm_mma(tile_x, tile_w, c13, 1, wave_t, wave_n, axis, group);
        __syncthreads();
    }

    dit_gemm_store(c00, out, t0 + 0 * kBaseT, row0, T, N, wave_t, group);
    dit_gemm_store(c01, out, t0 + 1 * kBaseT, row0, T, N, wave_t, group);
    dit_gemm_store(c02, out, t0 + 2 * kBaseT, row0, T, N, wave_t, group);
    dit_gemm_store(c03, out, t0 + 3 * kBaseT, row0, T, N, wave_t, group);
    dit_gemm_store(c10, out, t0 + 0 * kBaseT, row1, T, N, wave_t, group);
    dit_gemm_store(c11, out, t0 + 1 * kBaseT, row1, T, N, wave_t, group);
    dit_gemm_store(c12, out, t0 + 2 * kBaseT, row1, T, N, wave_t, group);
    dit_gemm_store(c13, out, t0 + 3 * kBaseT, row1, T, N, wave_t, group);
}

void dit_linear(const Tensor& x, const Weight& weight, Tensor& out, hipStream_t stream) {
    const int K = static_cast<int>(x.ne[0]);
    const int T = static_cast<int>(x.ne[1]);
    const int N = static_cast<int>(out.ne[0]);
    if (T <= 0 || K <= 0 || K % kGemmK != 0 || out.ne[1] != T || weight.k != K || weight.n != N ||
        weight.qdata == nullptr) {
        throw std::runtime_error("dit_linear: unsupported shape");
    }
    dim3 grid((N + kGemmM - 1) / kGemmM, (T + kGemmT - 1) / kGemmT);
    dit_gemm_kernel<<<grid, kGemmThreads, 0, stream>>>(
        static_cast<const hip_bfloat16*>(x.data), static_cast<const hip_bfloat16*>(weight.qdata),
        static_cast<hip_bfloat16*>(out.data), T, N, K);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::targets::qwen_image_21::detail
