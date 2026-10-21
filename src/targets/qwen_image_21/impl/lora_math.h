#pragma once

// BF16 helpers and one LoRA output element. Host and device share this so a denoise
// adds the same scale * (B @ A) the merge script writes, with the same round-to-nearest-even.
// B is (rows, rank) and A is (rank, cols), both row-major BF16. The weight element is (row, col).

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define NINFER_HD __host__ __device__
#else
#define NINFER_HD
#endif

NINFER_HD inline void ninfer_copy4(void* dst, const void* src) {
    const auto* in = static_cast<const unsigned char*>(src);
    auto* out      = static_cast<unsigned char*>(dst);
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
    out[3] = in[3];
}

NINFER_HD inline float ninfer_f32_from_bf16(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.F;
    ninfer_copy4(&value, &wide);
    return value;
}

NINFER_HD inline std::uint16_t ninfer_bf16_from_f32(float value) {
    std::uint32_t bits = 0;
    ninfer_copy4(&bits, &value);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

// weight + scale * dot(B[row], A[:, col]). A is addressed as a[k * cols + col].
NINFER_HD inline std::uint16_t ninfer_lora_element(std::uint16_t weight, const std::uint16_t* b_row,
                                                  const std::uint16_t* a, int cols, int rank, int col,
                                                  float scale) {
    float acc = 0.F;
    for (int k = 0; k < rank; ++k) {
        acc += ninfer_f32_from_bf16(b_row[k]) * ninfer_f32_from_bf16(a[static_cast<std::size_t>(k) * cols + col]);
    }
    return ninfer_bf16_from_f32(ninfer_f32_from_bf16(weight) + scale * acc);
}
