// Causal GQA for the Qwen3-VL text encoder (wave32 gfx1201). One block per (token, head, batch).
#include "causal_gqa.h"

#include <hip/hip_bfloat16.h>

#include <cfloat>
#include <cstdint>

namespace ninfer::targets::qwen3_vl::detail {
namespace {

template <int HeadDim, int GqaGroup>
__global__ void causal_gqa_kernel(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
                                  const hip_bfloat16* __restrict__ v, hip_bfloat16* __restrict__ out,
                                  int tokens, int heads, int kv_heads, float scale) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int batch = blockIdx.z;
    const int kvh   = head / GqaGroup;  // GQA: query head -> KV head
    const std::int64_t qstride = (std::int64_t)heads * HeadDim;
    const std::int64_t kstride = (std::int64_t)kv_heads * HeadDim;
    const std::int64_t qbase = (std::int64_t)batch * tokens * qstride +
                               (std::int64_t)token * qstride + (std::int64_t)head * HeadDim;
    const hip_bfloat16* kbase = k + (std::int64_t)batch * tokens * kstride + (std::int64_t)kvh * HeadDim;
    const hip_bfloat16* vbase = v + (std::int64_t)batch * tokens * kstride + (std::int64_t)kvh * HeadDim;
    const hip_bfloat16* qrow = q + qbase;
    hip_bfloat16* orow = out + qbase;

    __shared__ float qs[HeadDim];
    for (int i = threadIdx.x; i < HeadDim; i += 256) qs[i] = qrow[i];
    __syncthreads();

    // Pass 1: max and sum-exp over keys j <= token (causal triangle).
    float m = -FLT_MAX, l = 0.0F;
    for (int j = threadIdx.x; j <= token; j += 256) {
        float dot = 0.0F;
        for (int d = 0; d < HeadDim; d++) {
            dot += qs[d] * kbase[(std::int64_t)j * kstride + d];
        }
        dot *= scale;
        const float nm = fmaxf(m, dot);
        l = l * __expf(m - nm) + __expf(dot - nm);
        m = nm;
    }
    __shared__ float red[2][256 / 32];
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_down_sync(0xffffffffull, m, off));
        l += __shfl_down_sync(0xffffffffull, l, off);
    }
    if ((threadIdx.x & 31) == 0) {
        red[0][threadIdx.x / 32] = m;
        red[1][threadIdx.x / 32] = l;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mm = -FLT_MAX, ll = 0.0F;
        for (int w = 0; w < 256 / 32; w++) {
            mm = fmaxf(mm, red[0][w]);
            ll += red[1][w];
        }
        red[0][0] = mm;
        red[1][0] = ll;
    }
    __syncthreads();
    const float inv_l = 1.0F / red[1][0];

    // Pass 2: weighted sum of values over keys j <= token (causal triangle).
    float acc[(HeadDim + 255) / 256];
    for (int d = threadIdx.x; d < HeadDim; d += 256) acc[d / 256] = 0.0F;
    for (int j = 0; j <= token; j++) {
        float dot = 0.0F;
        for (int d = 0; d < HeadDim; d++) {
            dot += qs[d] * kbase[(std::int64_t)j * kstride + d];
        }
        const float w = __expf(dot * scale - red[0][0]) * inv_l;
        for (int d = threadIdx.x; d < HeadDim; d += 256) {
            acc[d / 256] += w * vbase[(std::int64_t)j * kstride + d];
        }
    }
    for (int d = threadIdx.x; d < HeadDim; d += 256) {
        orow[d] = hip_bfloat16(acc[d / 256]);
    }
}

} // namespace

void causal_gqa(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out, hipStream_t stream) {
    const int tokens   = q.ne[1];
    const int heads    = q.ne[2];
    const int batch    = q.ne[0];
    const int head_dim = q.ne[3];
    const int kv_heads = k.ne[2];
    const float scale  = 1.0F / sqrtf((float)head_dim);
    dim3 grid(tokens, heads, batch);
    dim3 block(256);
    if (head_dim == 128 && heads == 32 && kv_heads == 8) {
        causal_gqa_kernel<128, 4><<<grid, block, 0, stream>>>((const hip_bfloat16*)q.data,
                                                              (const hip_bfloat16*)k.data,
                                                              (const hip_bfloat16*)v.data,
                                                              (hip_bfloat16*)out.data, tokens, heads, kv_heads, scale);
    } else {
        throw std::runtime_error("causal_gqa: unsupported shape");
    }
}

} // namespace ninfer::targets::qwen3_vl::detail
