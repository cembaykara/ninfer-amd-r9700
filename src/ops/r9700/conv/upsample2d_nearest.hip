// 2x nearest-neighbor upsampling (NCHW, single image).
#include "ninfer/ops/upsample2d_nearest.h"

#include <hip/hip_bfloat16.h>

namespace ninfer::ops {
namespace {

constexpr int kTile = 8;

__global__ void upsample2d_nearest_kernel(const hip_bfloat16* __restrict__ in, int h, int w,
                                          hip_bfloat16* __restrict__ out) {
    const int oc = blockIdx.z;
    const int ox = blockIdx.x * kTile + threadIdx.x;
    const int oy = blockIdx.y * kTile + threadIdx.y;
    if (ox >= 2 * w || oy >= 2 * h) return;
    const int ix = ox / 2;
    const int iy = oy / 2;
    out[(std::int64_t)oc * 2 * h * 2 * w + (std::int64_t)oy * 2 * w + ox] =
        in[(std::int64_t)oc * h * w + (std::int64_t)iy * w + ix];
}

} // namespace

void upsample2d_nearest(const Tensor& in, Tensor& out, hipStream_t stream) {
    const int c = in.ne[0];
    const int h = in.ne[1];
    const int w = in.ne[2];
    dim3 block(kTile, kTile, 1);
    dim3 grid((2 * w + kTile - 1) / kTile, (2 * h + kTile - 1) / kTile, c);
    upsample2d_nearest_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const hip_bfloat16*>(in.data), h, w,
        reinterpret_cast<hip_bfloat16*>(out.data));
}

} // namespace ninfer::ops
