// 2D convolution, 3x3 kernel, stride 1, padding 1 (NCHW, single image). Direct implementation.
#include "ninfer/ops/conv2d_3x3.h"

#include <hip/hip_bfloat16.h>

namespace ninfer::ops {
namespace {

constexpr int kTile = 8;

__global__ void conv2d_3x3_kernel(const hip_bfloat16* __restrict__ in,
                                  const hip_bfloat16* __restrict__ weight,
                                  const hip_bfloat16* __restrict__ bias,
                                  int c_in, int h, int w, hip_bfloat16* __restrict__ out) {
    const int c_out = blockIdx.z;
    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x >= w || y >= h) return;

    float acc = float(bias[c_out]);
    const int c_out_base = c_out * c_in * 9;
    for (int ci = 0; ci < c_in; ++ci) {
        const hip_bfloat16* in_ci = in + (std::int64_t)ci * h * w;
        const hip_bfloat16* w_ci = weight + c_out_base + ci * 9;
        // 3x3 taps
        const int y0 = y - 1, y1 = y, y2 = y + 1;
        const int x0 = x - 1, x1 = x, x2 = x + 1;
        const bool y0ok = y0 >= 0 && y0 < h;
        const bool y1ok = y1 >= 0 && y1 < h;
        const bool y2ok = y2 >= 0 && y2 < h;
        const bool x0ok = x0 >= 0 && x0 < w;
        const bool x1ok = x1 >= 0 && x1 < w;
        const bool x2ok = x2 >= 0 && x2 < w;
        if (y0ok) {
            if (x0ok) acc += float(w_ci[0]) * float(in_ci[(std::int64_t)y0 * w + x0]);
            if (x1ok) acc += float(w_ci[1]) * float(in_ci[(std::int64_t)y0 * w + x1]);
            if (x2ok) acc += float(w_ci[2]) * float(in_ci[(std::int64_t)y0 * w + x2]);
        }
        if (y1ok) {
            if (x0ok) acc += float(w_ci[3]) * float(in_ci[(std::int64_t)y1 * w + x0]);
            if (x1ok) acc += float(w_ci[4]) * float(in_ci[(std::int64_t)y1 * w + x1]);
            if (x2ok) acc += float(w_ci[5]) * float(in_ci[(std::int64_t)y1 * w + x2]);
        }
        if (y2ok) {
            if (x0ok) acc += float(w_ci[6]) * float(in_ci[(std::int64_t)y2 * w + x0]);
            if (x1ok) acc += float(w_ci[7]) * float(in_ci[(std::int64_t)y2 * w + x1]);
            if (x2ok) acc += float(w_ci[8]) * float(in_ci[(std::int64_t)y2 * w + x2]);
        }
    }
    out[(std::int64_t)c_out * h * w + (std::int64_t)y * w + x] = hip_bfloat16(acc);
}

} // namespace

void conv2d_3x3(const Tensor& in, const Tensor& weight, const Tensor& bias, Tensor& out,
                hipStream_t stream) {
    const int c_in = in.ne[0];
    const int h    = in.ne[1];
    const int w    = in.ne[2];
    const int c_out = weight.ne[0];
    dim3 block(kTile, kTile, 1);
    dim3 grid((w + kTile - 1) / kTile, (h + kTile - 1) / kTile, c_out);
    conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const hip_bfloat16*>(in.data),
        reinterpret_cast<const hip_bfloat16*>(weight.data),
        reinterpret_cast<const hip_bfloat16*>(bias.data),
        c_in, h, w, reinterpret_cast<hip_bfloat16*>(out.data));
}

} // namespace ninfer::ops
