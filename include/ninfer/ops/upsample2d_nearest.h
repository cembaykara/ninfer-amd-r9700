#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::ops {

/**
 * 2x nearest-neighbor upsampling (NCHW, single image). in [C, H, W] -> out [C, 2H, 2W].
 * out[c, 2y, 2x] = out[c, 2y+1, 2x] = out[c, 2y, 2x+1] = out[c, 2y+1, 2x+1] = in[c, y, x].
 * This is the "nearest-exact" interpolation the VAE upsampler uses before its 3x3 conv.
 */
void upsample2d_nearest(const Tensor& in, Tensor& out, hipStream_t stream);

} // namespace ninfer::ops
