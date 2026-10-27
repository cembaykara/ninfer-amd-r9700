// WMMA attention for the Qwen-Image-2.1 DiT (wave32 gfx1201).
//
// Mask: HF block-causal — text causal; each condition image its own bidirectional
// block; target bidirectional and sees the full prefix.
#include "dit_mha.h"

#include "core/device.h"

#include <hip/hip_bfloat16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen_image_21::detail {
namespace {

constexpr int kHeadDim       = 128;
constexpr int kMatrixTile    = 16;
constexpr int kWave          = 32;
constexpr int kWavesPerBlock = 8;
constexpr int kQueryTile     = kWavesPerBlock * kMatrixTile;
constexpr int kThreads       = kWave * kWavesPerBlock;
constexpr int kKeyTile       = 32;
constexpr int kKeySubtiles   = kKeyTile / kMatrixTile;
constexpr int kFeatureTiles  = kHeadDim / kMatrixTile;
constexpr int kChunk         = 8;
constexpr int kChunks        = kHeadDim / kChunk;
constexpr int kKeyPitch      = kHeadDim + kChunk;
constexpr int kValuePitch    = kKeyTile + kChunk;
constexpr int kMaxImg        = 10;
constexpr float kScale       = 0.08838834764831845F; // 1/sqrt(128)
constexpr float kLog2Scale   = kScale * 1.44269504088896340736F;

using Bf16Fragment = short __attribute__((ext_vector_type(8)));
using F32Fragment  = float __attribute__((ext_vector_type(8)));

static_assert(kHeadDim % kMatrixTile == 0);

__device__ __forceinline__ short bf16_bits(float value) {
    return static_cast<short>(hip_bfloat16(value).data);
}

struct ImgBlocks {
    int n;
    int start[kMaxImg];
    int end[kMaxImg];
    int target_start;
};

__global__ __launch_bounds__(kThreads) void dit_mha_kernel(const hip_bfloat16* __restrict__ q,
                                                           const hip_bfloat16* __restrict__ k,
                                                           const hip_bfloat16* __restrict__ v,
                                                           hip_bfloat16* __restrict__ out, int seq,
                                                           ImgBlocks blocks, std::int64_t stride) {
    __shared__ __attribute__((aligned(16))) short keys[kKeyTile * kKeyPitch];
    __shared__ __attribute__((aligned(16))) short values_t[kHeadDim * kValuePitch];

    const int tid   = static_cast<int>(threadIdx.x);
    const int wave  = tid / kWave;
    const int lane  = tid % kWave;
    const int half  = lane / kMatrixTile;
    const int axis  = lane % kMatrixTile;
    const int head  = static_cast<int>(blockIdx.y);
    const int q0    = static_cast<int>(blockIdx.x) * kQueryTile;
    const int query = q0 + wave * kMatrixTile + axis;
    const bool query_valid = query < seq;

    for (int index = tid; index < kKeyTile; index += kThreads) {
        *reinterpret_cast<uint4*>(&keys[index * kKeyPitch + kHeadDim]) = uint4{};
    }

    Bf16Fragment query_fragment[kFeatureTiles];
#pragma unroll
    for (int feature_tile = 0; feature_tile < kFeatureTiles; ++feature_tile) {
        const int chunk              = feature_tile * 2 + half;
        query_fragment[feature_tile] = Bf16Fragment{};
        if (query_valid) {
            query_fragment[feature_tile] = *reinterpret_cast<const Bf16Fragment*>(
                q + static_cast<std::int64_t>(query) * stride + head * kHeadDim + chunk * kChunk);
        }
    }

    int q_block_start = -1;
    int q_block_end   = -1;
    if (query_valid) {
        if (query >= blocks.target_start) {
            q_block_start = blocks.target_start;
            q_block_end   = seq;
        } else {
            for (int i = 0; i < blocks.n; ++i) {
                if (query >= blocks.start[i] && query < blocks.end[i]) {
                    q_block_start = blocks.start[i];
                    q_block_end   = blocks.end[i];
                    break;
                }
            }
        }
    }

    const int q_last = min(q0 + kQueryTile, seq) - 1;
    const int kv_end = q_last >= blocks.target_start ? seq : q_last + 1;

    float maximum     = -INFINITY;
    float denominator = 0.0F;
    F32Fragment attended[kFeatureTiles];
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < kFeatureTiles; ++dimension_tile) {
        attended[dimension_tile] = F32Fragment{};
    }

    for (int key_base = 0; key_base < kv_end; key_base += kKeyTile) {
        __syncthreads();
        for (int index = tid; index < kKeyTile * kChunks; index += kThreads) {
            const int key_local = index / kChunks;
            const int chunk     = index - key_local * kChunks;
            const int key       = key_base + key_local;
            uint4 key_bits{};
            uint4 value_bits{};
            if (key < seq) {
                key_bits = *reinterpret_cast<const uint4*>(
                    k + static_cast<std::int64_t>(key) * stride + head * kHeadDim + chunk * kChunk);
                value_bits = *reinterpret_cast<const uint4*>(
                    v + static_cast<std::int64_t>(key) * stride + head * kHeadDim + chunk * kChunk);
            }
            *reinterpret_cast<uint4*>(&keys[key_local * kKeyPitch + chunk * kChunk]) = key_bits;
            const short* value_elements = reinterpret_cast<const short*>(&value_bits);
#pragma unroll
            for (int element = 0; element < kChunk; ++element) {
                values_t[(chunk * kChunk + element) * kValuePitch + key_local] =
                    value_elements[element];
            }
        }
        __syncthreads();

        F32Fragment score[kKeySubtiles];
#pragma unroll
        for (int key_tile = 0; key_tile < kKeySubtiles; ++key_tile) {
            score[key_tile] = F32Fragment{};
#pragma unroll
            for (int feature_tile = 0; feature_tile < kFeatureTiles; ++feature_tile) {
                const Bf16Fragment key_fragment = *reinterpret_cast<const Bf16Fragment*>(
                    &keys[(key_tile * kMatrixTile + axis) * kKeyPitch + feature_tile * kMatrixTile +
                          half * kChunk]);
                score[key_tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    key_fragment, query_fragment[feature_tile], score[key_tile]);
            }
        }

        float tile_maximum = -INFINITY;
#pragma unroll
        for (int key_tile = 0; key_tile < kKeySubtiles; ++key_tile) {
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const int key = key_base + key_tile * kMatrixTile + half * kChunk + item;
                const bool same_block =
                    q_block_start >= 0 && key >= q_block_start && key < q_block_end;
                // HF: (q_idx >= kv_idx) | same_image_block
                const bool allowed = key < seq && (key <= query || same_block);
                const float scaled = allowed ? score[key_tile][item] * kLog2Scale : -INFINITY;
                score[key_tile][item] = scaled;
                tile_maximum          = fmaxf(tile_maximum, scaled);
            }
        }
        tile_maximum = fmaxf(tile_maximum, __shfl_xor(tile_maximum, kMatrixTile, kWave));
        const float next_maximum = fmaxf(maximum, tile_maximum);
        const float origin = next_maximum == -INFINITY ? 0.0F : next_maximum;
        const float alpha  = exp2f(maximum - origin);
        float tile_sum     = 0.0F;
#pragma unroll
        for (int key_tile = 0; key_tile < kKeySubtiles; ++key_tile) {
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                float probability = exp2f(score[key_tile][item] - origin);
                if (!(probability >= 0.0F)) probability = 0.0F;
                score[key_tile][item] = probability;
                tile_sum += probability;
            }
        }
        tile_sum += __shfl_xor(tile_sum, kMatrixTile, kWave);
        denominator = denominator * alpha + tile_sum;
        maximum     = next_maximum;
#pragma unroll
        for (int dimension_tile = 0; dimension_tile < kFeatureTiles; ++dimension_tile) {
            attended[dimension_tile] *= alpha;
        }

#pragma unroll
        for (int key_tile = 0; key_tile < kKeySubtiles; ++key_tile) {
            Bf16Fragment probability_high;
            Bf16Fragment probability_low;
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const float probability = score[key_tile][item];
                const hip_bfloat16 high(probability);
                probability_high[item] = static_cast<short>(high.data);
                probability_low[item]  = bf16_bits(probability - static_cast<float>(high));
            }
#pragma unroll
            for (int dimension_tile = 0; dimension_tile < kFeatureTiles; ++dimension_tile) {
                const Bf16Fragment value_fragment = *reinterpret_cast<const Bf16Fragment*>(
                    &values_t[(dimension_tile * kMatrixTile + axis) * kValuePitch +
                              key_tile * kMatrixTile + half * kChunk]);
                attended[dimension_tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    value_fragment, probability_high, attended[dimension_tile]);
                attended[dimension_tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    value_fragment, probability_low, attended[dimension_tile]);
            }
        }
    }

    if (!query_valid) { return; }
    const float inverse = denominator > 0.0F ? 1.0F / denominator : 0.0F;
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < kFeatureTiles; ++dimension_tile) {
        const int chunk = dimension_tile * 2 + half;
        Bf16Fragment packed;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            packed[item] = bf16_bits(attended[dimension_tile][item] * inverse);
        }
        *reinterpret_cast<Bf16Fragment*>(out + static_cast<std::int64_t>(query) * stride +
                                         head * kHeadDim + chunk * kChunk) = packed;
    }
}

} // namespace

void dit_mha(const Tensor& q, const Tensor& k, const Tensor& v, const int* img_block_starts,
             const int* img_block_ends, int n_img, int target_start, Tensor& out,
             hipStream_t stream) {
    if (q.ne[0] != kHeadDim || k.ne[0] != kHeadDim || v.ne[0] != kHeadDim || out.ne[0] != kHeadDim) {
        throw std::runtime_error("dit_mha: head_dim must be 128");
    }
    if (n_img < 0 || n_img > kMaxImg) {
        throw std::runtime_error("dit_mha: n_img out of range");
    }
    const int heads = static_cast<int>(q.ne[1]);
    const int seq   = static_cast<int>(q.ne[2]);
    if (seq <= 0) { return; }
    if (target_start < 0 || target_start > seq) {
        throw std::runtime_error("dit_mha: invalid target_start");
    }
    ImgBlocks blocks{};
    blocks.n            = n_img;
    blocks.target_start = target_start;
    for (int i = 0; i < n_img; ++i) {
        if (img_block_starts == nullptr || img_block_ends == nullptr ||
            img_block_starts[i] < 0 || img_block_ends[i] < img_block_starts[i] ||
            img_block_ends[i] > target_start) {
            throw std::runtime_error("dit_mha: invalid image block");
        }
        blocks.start[i] = img_block_starts[i];
        blocks.end[i]   = img_block_ends[i];
    }
    const std::int64_t stride = static_cast<std::int64_t>(heads) * kHeadDim;
    dim3 grid((seq + kQueryTile - 1) / kQueryTile, heads);
    dit_mha_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const hip_bfloat16*>(q.data), static_cast<const hip_bfloat16*>(k.data),
        static_cast<const hip_bfloat16*>(v.data), static_cast<hip_bfloat16*>(out.data), seq, blocks,
        stride);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::targets::qwen_image_21::detail
