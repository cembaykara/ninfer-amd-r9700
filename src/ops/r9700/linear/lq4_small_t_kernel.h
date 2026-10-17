#pragma once

// LQ4 small-T projection kernel (T <= 16 per token tile; kTiles = 2 serves T 17..32): one CTA
// context per 16 rows of the first or the second projection. Shared by the Linear launcher
// (lq4_linear.hip) and the persistent decode kernel.
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/lq4_linear.h"
#include "ops/r9700/linear/lq4_pipeline_impl.h"
#include "ops/r9700/persistent/persistent_cta.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear::lq4_kernels {

using lq4::decode8;
using lq4::F32x8;
using lq4::I32x2;
using U32x4 = std::uint32_t __attribute__((ext_vector_type(4)));

inline constexpr std::uint16_t kCanonicalBf16QuietNan = 0x7FC0U;

// Branch-free round-to-nearest-even to BF16 (identical to hip_bfloat16 for every non-NaN value;
// NaN becomes the canonical quiet NaN).
__device__ __forceinline__ std::uint16_t round_bf16(float value) {
    const std::uint32_t bits    = __float_as_uint(value);
    const std::uint32_t rounded = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U;
    return value != value ? kCanonicalBf16QuietNan : static_cast<std::uint16_t>(rounded);
}

__device__ __forceinline__ float bf16_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

// One launch projects `weight` (and, small T, `second` in CTAs past weight.rows / 16).
struct KernelArgs {
    Lq4Weight weight{};
    Lq4Output output{};
    Lq4Weight second{};
    Lq4Output second_output{};
    const std::uint8_t* activation = nullptr;
    const float* token_scales      = nullptr;
    const std::uint32_t* status    = nullptr;
    std::uint32_t tokens           = 0;
};

// Element (token, row) of one published output.
__device__ __forceinline__ std::uint16_t* output_at(const Lq4Weight& weight,
                                                    const Lq4Output& output, std::uint32_t token,
                                                    std::uint32_t row) {
    const std::uint32_t rows = weight.rows, split = output.split;
    const bool leading        = split == 0U || row < split;
    const std::uint32_t width = split == 0U ? rows : (leading ? split : rows - split);
    auto* base = reinterpret_cast<std::uint16_t*>(leading ? output.leading : output.trailing);
    return base + static_cast<std::size_t>(token) * width + (leading ? row : row - split);
}

__device__ __forceinline__ std::uint16_t* output_at(const KernelArgs& a, std::uint32_t token,
                                                    std::uint32_t row) {
    return output_at(a.weight, a.output, token, row);
}

// BF16(silu(g) * u) of the BF16-rounded gate and up projections. The hardware exponential and
// reciprocal (a few FP32 ulps before the BF16 rounding) keep the fully unrolled prefill epilogue
// compact; accurate expf/division sequences tripled the kernel's code and slowed its main loop.
__device__ __forceinline__ std::uint16_t silu_pair_value(float gate, float up) {
    const float g = bf16_float(round_bf16(gate));
    const float u = bf16_float(round_bf16(up));
    return round_bf16(g * __builtin_amdgcn_rcpf(1.0F + __expf(-g)) * u);
}

// One token: 32 rows/CTA, four K quarters, four load steps in flight. Both the ordinary
// launcher and persistent lowering execute this same body and preserve its reduction order.
inline constexpr std::uint32_t kGemvSplit = 4U, kGemvSteps = 4U;

struct T1Shared {
    uint2 table[256];
    float partial[kGemvSplit][32];
};

template <bool kAccumulate, bool kSiluPair, class Cta>
__device__ __forceinline__ void t1_body(const Cta& cta, T1Shared& shared, const KernelArgs& a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t first_blocks = a.weight.rows / 32U;
    const bool second                = cta.block().x >= first_blocks;
    const Lq4Weight weight           = second ? a.second : a.weight;
    const Lq4Output output           = second ? a.second_output : a.output;
    const std::uint32_t row_base = (second ? cta.block().x - first_blocks : cta.block().x) * 32U;
    lq4::load_table(shared.table, cta.thread(), 32U * kGemvSplit);
    const std::uint32_t lane = cta.thread() & 31U, wave = cta.thread() >> 5U;
    const std::uint32_t row     = row_base + lane;
    const std::uint32_t columns = weight.columns, steps_total = columns / 64U;
    const std::uint32_t steps    = steps_total / kGemvSplit;
    const std::size_t first_tile = static_cast<std::size_t>(row / 16U) * steps_total + wave * steps;
    const auto* codes  = persistent::global(weight.codes) + first_tile * 512U + (row % 16U) * 16U;
    const auto* groups = persistent::global(weight.groups) + first_tile * 32U + row % 16U;
    const auto* activation =
        persistent::global(a.activation) + static_cast<std::size_t>(wave) * steps * 64U;

    struct Batch {
        uint4 c[kGemvSteps][2];
        std::uint32_t g[kGemvSteps][2];
        uint4 x[kGemvSteps][4];
    };

    const auto load = [&](std::uint32_t step) {
        Batch value;
#    pragma unroll
        for (std::uint32_t i = 0; i < kGemvSteps; ++i) {
            const auto* tile  = codes + static_cast<std::size_t>(step + i) * 512U;
            value.c[i][0]     = persistent::load16(tile);
            value.c[i][1]     = persistent::load16(tile + 256U);
            const auto* group = groups + static_cast<std::size_t>(step + i) * 32U;
            value.g[i][0]     = group[0];
            value.g[i][1]     = group[16];
            const auto* x     = activation + static_cast<std::size_t>(step + i) * 64U;
            value.x[i][0]     = persistent::load16(x + 0U);
            value.x[i][1]     = persistent::load16(x + 16U);
            value.x[i][2]     = persistent::load16(x + 32U);
            value.x[i][3]     = persistent::load16(x + 48U);
        }
        return value;
    };
    float sum          = 0.0F;
    const auto consume = [&](const Batch& value) {
#    pragma unroll
        for (std::uint32_t i = 0; i < kGemvSteps; ++i) {
#    pragma unroll
            for (std::uint32_t half = 0; half < 2U; ++half) {
                const uint2 book = shared.table[value.g[i][half]];
                const uint4 word = value.c[i][half];
                const uint2 w0 = decode8(word.x, book), w1 = decode8(word.y, book);
                const uint2 w2 = decode8(word.z, book), w3 = decode8(word.w, book);
                const uint4 x0 = value.x[i][2U * half], x1 = value.x[i][2U * half + 1U];
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w0.x),
                                                        static_cast<int>(x0.x), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w0.y),
                                                        static_cast<int>(x0.y), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w1.x),
                                                        static_cast<int>(x0.z), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w1.y),
                                                        static_cast<int>(x0.w), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w2.x),
                                                        static_cast<int>(x1.x), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w2.y),
                                                        static_cast<int>(x1.y), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w3.x),
                                                        static_cast<int>(x1.z), sum);
                sum = __builtin_amdgcn_dot4_f32_fp8_fp8(static_cast<int>(w3.y),
                                                        static_cast<int>(x1.w), sum);
            }
        }
    };
    Batch current = load(0U);
    cta.sync(); // codebook
    for (std::uint32_t step = kGemvSteps; step < steps; step += kGemvSteps) {
        const Batch next = load(step);
        consume(current);
        current = next;
    }
    consume(current);
    shared.partial[wave][lane] = sum;
    cta.sync();
    if (wave != 0U) return;
    float total = shared.partial[0][lane];
#    pragma unroll
    for (std::uint32_t split = 1; split < kGemvSplit; ++split) total += shared.partial[split][lane];
    const float y =
        total * persistent::global(weight.scales)[row] * persistent::global(a.token_scales)[0];
    const bool poison = persistent::global(a.status)[0] != Fp8ActivationOk;
    if constexpr (kSiluPair) {
        // Rows 16 b + i (gate) and 16 b + 8 + i (up) sit in lanes l and l ^ 8.
        const float partner = __shfl_xor(y, 8, 32);
        if ((lane & 8U) == 0U) {
            auto* out = persistent::global(reinterpret_cast<std::uint16_t*>(output.leading)) +
                        row_base / 2U + (lane / 16U) * 8U + lane % 8U;
            *out      = poison ? kCanonicalBf16QuietNan : silu_pair_value(y, partner);
        }
    } else {
        std::uint16_t* out = persistent::global(output_at(weight, output, 0U, row));
        if (poison) {
            *out = kCanonicalBf16QuietNan;
        } else if constexpr (kAccumulate) {
            *out = round_bf16(bf16_float(*out) + bf16_float(round_bf16(y)));
        } else {
            *out = round_bf16(y);
        }
    }
#else
    (void)cta;
    (void)shared;
    (void)a;
#endif
}

template <bool kAccumulate, bool kSiluPair>
__global__ __launch_bounds__(32U * kGemvSplit) void lq4_t1_kernel(KernelArgs a) {
    __shared__ T1Shared shared;
    t1_body<kAccumulate, kSiluPair>(persistent::LaunchCta{}, shared, a);
}

template <bool kSiluPair, std::uint32_t kSplit, std::uint32_t kTiles>
struct SmallTShared {
    uint2 table[256];
    float partial[kSplit][kTiles][8][32];
    float staged[kSiluPair ? 16 : 1][16U * kTiles];
};

template <bool kAccumulate, bool kSiluPair, std::uint32_t kSplit, std::uint32_t kSteps,
          std::uint32_t kTiles, class Cta>
__device__ __forceinline__ void
small_t_body(const Cta& cta, SmallTShared<kSiluPair, kSplit, kTiles>& shared, const KernelArgs& a) {
    const std::uint32_t first_blocks = a.weight.rows / 16U;
    const std::uint32_t block        = cta.block().x;
    const bool second                = block >= first_blocks;
    // By value: the CTA-uniform selection stays in scalar registers.
    const Lq4Weight weight       = second ? a.second : a.weight;
    const Lq4Output output       = second ? a.second_output : a.output;
    const std::uint32_t row_base = (second ? block - first_blocks : block) * 16U;
    lq4::load_table(shared.table, cta.thread(), 32U * kSplit);
    const auto publish = [&](std::uint32_t token, std::uint32_t row, float value) {
        if constexpr (kSiluPair) {
            shared.staged[row - row_base][token] = value;
            return;
        }
        auto* out = persistent::global(output_at(weight, output, token, row));
        if (persistent::global(a.status)[token] != Fp8ActivationOk) {
            *out = kCanonicalBf16QuietNan;
        } else if constexpr (kAccumulate) {
            *out = round_bf16(bf16_float(*out) + bf16_float(round_bf16(value)));
        } else {
            *out = round_bf16(value);
        }
    };
    lq4::small_t_rows<kSplit, kSteps, kTiles>(cta, weight, row_base, a.activation, a.token_scales,
                                              a.tokens, shared.table, shared.partial, publish);
    if constexpr (kSiluPair) {
        cta.sync();
        const std::uint32_t features = weight.rows / 2U;
        for (std::uint32_t index = cta.thread(); index < 8U * a.tokens; index += cta.threads()) {
            const std::uint32_t feature = index % 8U, token = index / 8U;
            auto* out = persistent::global(reinterpret_cast<std::uint16_t*>(output.leading)) +
                        static_cast<std::size_t>(token) * features + row_base / 2U + feature;
            *out      = persistent::global(a.status)[token] != Fp8ActivationOk
                            ? kCanonicalBf16QuietNan
                            : silu_pair_value(shared.staged[feature][token],
                                              shared.staged[feature + 8U][token]);
        }
    }
}

template <bool kAccumulate, bool kSiluPair, std::uint32_t kSplit, std::uint32_t kSteps,
          std::uint32_t kTiles>
__global__ __launch_bounds__(32U * kSplit) void lq4_small_t_kernel(KernelArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ SmallTShared<kSiluPair, kSplit, kTiles> shared;
    small_t_body<kAccumulate, kSiluPair, kSplit, kSteps, kTiles>(persistent::LaunchCta{}, shared,
                                                                 a);
#else
    (void)a;
#endif
}

} // namespace ninfer::ops::r9700::linear::lq4_kernels
