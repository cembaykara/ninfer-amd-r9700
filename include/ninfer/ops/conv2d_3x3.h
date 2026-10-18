#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::ops {

/**
 * 2D convolution, 3x3 kernel, stride 1, padding 1 (same size). NCHW layout, single image.
 *
 *   in:     BF16 [C_in, H, W]
 *   weight: BF16 [C_out, C_in, 3, 3]
 *   bias:   BF16 [C_out] (optional; pass a zero-initialized tensor to skip)
 *   out:    BF16 [C_out, H, W]
 *
 * out[c_out, y, x] = bias[c_out] + sum_{c_in, ky, kx} weight[c_out, c_in, ky, kx]
 *                    * in[c_in, y+ky-1, x+kx-1]   (zero outside [0,H)x[0,W))
 *
 * Direct (non-GEMM) implementation: one thread per (c_out, y, x), accumulating over
 * C_in*9 taps in FP32. Correct for the VAE decode (a one-time cost, not in the denoise loop).
 */
void conv2d_3x3(const Tensor& in, const Tensor& weight, const Tensor& bias, Tensor& out,
                hipStream_t stream);

} // namespace ninfer::ops
