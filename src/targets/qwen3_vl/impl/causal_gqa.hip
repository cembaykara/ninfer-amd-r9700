// Causal GQA for the Qwen3-VL text encoder (wave32 gfx1201). One block per (token, head).
// q/out are {head_dim, heads, tokens}; k/v are {head_dim, kv_heads, tokens}.
#include "causal_gqa.h"

#include <hip/hip_bfloat16.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_vl::detail {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps   = kThreads / 32;

template <bool IsMax>
__device__ float block_reduce(float value, float* scratch) {
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffffull, value, off);
        value             = IsMax ? fmaxf(value, other) : value + other;
    }
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x / 32] = value; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = scratch[0];
        for (int w = 1; w < kWarps; w++) { r = IsMax ? fmaxf(r, scratch[w]) : r + scratch[w]; }
        scratch[0] = r;
    }
    __syncthreads();
    const float r = scratch[0];
    __syncthreads();
    return r;
}

template <int HeadDim, int GqaGroup>
__global__ void causal_gqa_kernel(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
                                  const hip_bfloat16* __restrict__ v, hip_bfloat16* __restrict__ out,
                                  int heads, int kv_heads, float scale) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int kvh   = head / GqaGroup;
    const std::int64_t qstride = (std::int64_t)heads * HeadDim;
    const std::int64_t kstride = (std::int64_t)kv_heads * HeadDim;
    const hip_bfloat16* qrow  = q + (std::int64_t)token * qstride + (std::int64_t)head * HeadDim;
    const hip_bfloat16* kbase = k + (std::int64_t)kvh * HeadDim;
    const hip_bfloat16* vbase = v + (std::int64_t)kvh * HeadDim;
    hip_bfloat16* orow        = out + (std::int64_t)token * qstride + (std::int64_t)head * HeadDim;

    __shared__ float qs[HeadDim];
    __shared__ float scratch[kWarps];
    for (int i = threadIdx.x; i < HeadDim; i += kThreads) { qs[i] = (float)qrow[i]; }
    __syncthreads();

    auto score = [&](int j) {
        float dot = 0.0F;
        for (int d = 0; d < HeadDim; d++) { dot += qs[d] * (float)kbase[(std::int64_t)j * kstride + d]; }
        return dot * scale;
    };

    // Softmax over keys j <= token: the block-wide max first, then the sum against that max.
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j <= token; j += kThreads) { m = fmaxf(m, score(j)); }
    m = block_reduce<true>(m, scratch);
    float l = 0.0F;
    for (int j = threadIdx.x; j <= token; j += kThreads) { l += expf(score(j) - m); }
    l                 = block_reduce<false>(l, scratch);
    const float inv_l = 1.0F / l;

    for (int d = threadIdx.x; d < HeadDim; d += kThreads) {
        float acc = 0.0F;
        for (int j = 0; j <= token; j++) {
            acc += expf(score(j) - m) * inv_l * (float)vbase[(std::int64_t)j * kstride + d];
        }
        orow[d] = hip_bfloat16(acc);
    }
}

} // namespace

void causal_gqa(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out, hipStream_t stream) {
    const int head_dim = q.ne[0];
    const int heads    = q.ne[1];
    const int tokens   = q.ne[2];
    const int kv_heads = k.ne[1];
    const float scale  = 1.0F / sqrtf((float)head_dim);
    if (head_dim == 128 && heads == 32 && kv_heads == 8 && k.ne[2] == tokens && v.ne[2] == tokens) {
        causal_gqa_kernel<128, 4><<<dim3(tokens, heads), dim3(kThreads), 0, stream>>>(
            (const hip_bfloat16*)q.data, (const hip_bfloat16*)k.data, (const hip_bfloat16*)v.data,
            (hip_bfloat16*)out.data, heads, kv_heads, scale);
    } else {
        throw std::runtime_error("causal_gqa: unsupported shape");
    }
}

} // namespace ninfer::targets::qwen3_vl::detail
