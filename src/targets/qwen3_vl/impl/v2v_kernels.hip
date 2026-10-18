// v2v glue kernels (wave32 gfx1201).
#include "v2v_kernels.h"

#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen3_vl::detail {
namespace {

constexpr int kThreads   = 256;
constexpr int kPerThread = 4096 / kThreads;  // 16 elements per thread

__global__ void visual_merge_kernel(const hip_bfloat16* __restrict__ text,
                                    const hip_bfloat16* __restrict__ visual_src,
                                    int visual_start, int visual_count, int seq, int hidden,
                                    hip_bfloat16* __restrict__ out) {
    int pos = blockIdx.x;
    if (pos >= seq) return;
    const bool is_visual = (pos >= visual_start && pos < visual_start + visual_count);
    const hip_bfloat16* src = is_visual ? (visual_src + (pos - visual_start) * hidden)
                                        : (text + pos * hidden);
    hip_bfloat16* dst = out + pos * hidden;
    // copy kPerThread (16) elements as two 16-byte (uint4) vectors
    const int base = threadIdx.x * kPerThread;
    reinterpret_cast<uint4*>(dst + base)[0]      = reinterpret_cast<const uint4*>(src + base)[0];
    reinterpret_cast<uint4*>(dst + base)[1]      = reinterpret_cast<const uint4*>(src + base)[1];
}

__global__ void deepstack_inject_kernel(hip_bfloat16* __restrict__ hidden,
                                        const hip_bfloat16* __restrict__ feature,
                                        int visual_start, int visual_count, int hidden_dim) {
    int pos = blockIdx.x;
    if (pos < visual_start || pos >= visual_start + visual_count) return;
    hip_bfloat16* dst = hidden + pos * hidden_dim;
    const hip_bfloat16* src = feature + (pos - visual_start) * hidden_dim;
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid * kPerThread; i < (tid + 1) * kPerThread; i += kPerThread) {
        dst[i] = __hadd(dst[i], src[i]);
    }
}

} // namespace

void visual_merge(const hip_bfloat16* text_embeds, const hip_bfloat16* visual_embeds,
                  int visual_start, int visual_count, int seq, int hidden,
                  hip_bfloat16* out, hipStream_t stream) {
    dim3 grid(seq);
    visual_merge_kernel<<<grid, kThreads, 0, stream>>>(text_embeds, visual_embeds, visual_start,
                                                       visual_count, seq, hidden, out);
}

void deepstack_inject(hip_bfloat16* hidden, const hip_bfloat16* feature,
                      int visual_start, int visual_count, int seq, int hidden_dim,
                      hipStream_t stream) {
    dim3 grid(seq);
    deepstack_inject_kernel<<<grid, kThreads, 0, stream>>>(hidden, feature, visual_start,
                                                           visual_count, hidden_dim);
}

} // namespace ninfer::targets::qwen3_vl::detail
