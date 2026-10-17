#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::gdn_recurrence {

// Private chunked (WY) implementation of the normalized ordinary recurrence for widths
// >= kChunkedMinimumWidth (one masked chunk already beats the sequential kernel at T16). One
// workgroup per value head walks 64-token chunks: per chunk it forms A = strictly lower beta_i
// exp(b_i - b_j) k_i.k_j, the diagonal blocks of (I + A)^-1 in FP32, Z = beta (V - Gamma K S^T),
// the chunk's corrected values V' by block forward substitution, O = scale (Gamma q) S^T + (causal
// decayed q.k) V', and S <- Gamma_C S + (K exp(b_C - b))^T V'. Products run as FP16 WMMA with FP32
// accumulation; the state stays FP32 in the accumulators. A partial final chunk is masked (zero
// q/k/v/beta/g past the width). At T >= 256, chunk-parallel preparation shares normalized keys and
// FP32 QK/KK products among the three value heads, using explicit caller-owned workspace. State may
// alias state_out.
inline constexpr std::int32_t kChunkedMinimumWidth  = 16;
inline constexpr std::int32_t kPreparedMinimumWidth = 256;

[[nodiscard]] constexpr std::size_t prefill_workspace_bytes(std::int32_t width) noexcept {
    return width < kPreparedMinimumWidth ? 0
                                         : static_cast<std::size_t>((width + 63) / 64) * 16 *
                                               (64 * 128 * sizeof(_Float16) + 64 * sizeof(float) +
                                                20 * 16 * 16 * sizeof(float));
}

[[nodiscard]] hipError_t prefill_chunked(const hip_bfloat16* query, const hip_bfloat16* key,
                                         const hip_bfloat16* value, const float* g,
                                         const float* beta, const float* state_in, float* state_out,
                                         hip_bfloat16* output, std::int32_t width, float scale,
                                         void* workspace, hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::gdn_recurrence
