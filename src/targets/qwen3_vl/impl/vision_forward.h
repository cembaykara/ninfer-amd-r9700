#pragma once

// Qwen3-VL vision (ViT) encoder for the Qwen-Image text encoder target.
// Encodes an input image's patches into merged visual tokens [4096, V] plus the three
// deepstack features [4096, V] (from ViT layers 8/16/24) that the text stack injects at
// layers 0/1/2. Reuses shared ops: linear, add_bias, layer_norm, gelu, rope (Vision 2-D),
// vision_attention, vision_pos_embed_add, residual_add.
#include "config.h"
#include "core/tensor.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen3_vl::detail {

struct VisionConfig {
    static constexpr int hidden      = 1152;
    static constexpr int layers      = 27;
    static constexpr int heads       = 16;
    static constexpr int head_dim    = 72;          // hidden / heads
    static constexpr int intermediate = 4304;
    static constexpr int patch_dim   = 3 * 2 * 16 * 16; // 1152 (in_channels*temporal*patch*patch)
    static constexpr int merge_unit  = 4;           // spatial_merge_size^2
    static constexpr int merger_hidden = hidden * merge_unit; // 4608
    static constexpr int out_hidden  = 4096;
    static constexpr int pos_embed   = 2304;        // 48x48 table
    static constexpr int rotary_dim  = 72;
    static constexpr float rope_theta = 10000.0F;
    static constexpr float eps       = 1.0e-6F;
    static constexpr int deepstack_layers = 3;
    static constexpr int deepstack_index[3] = {8, 16, 24};
};

// The ViT weights, bound from the .ninfer artifact.
struct VisionWeights {
    Weight patch_embed;                        // [1152, 1536] (flattened conv: 3*2*16*16 patch features)
    Tensor patch_embed_bias;                   // [1152]
    Tensor pos_embed;                          // [2304, 1152]
    Tensor norm1_w[VisionConfig::layers];      // [1152]
    Tensor norm1_b[VisionConfig::layers];      // [1152]
    Weight qkv[VisionConfig::layers];          // [3456, 1152]
    Tensor qkv_bias[VisionConfig::layers];     // [3456]
    Weight proj[VisionConfig::layers];         // [1152, 1152]
    Tensor proj_bias[VisionConfig::layers];    // [1152]
    Tensor norm2_w[VisionConfig::layers];      // [1152]
    Tensor norm2_b[VisionConfig::layers];      // [1152]
    Weight fc1[VisionConfig::layers];          // [4304, 1152]
    Tensor fc1_bias[VisionConfig::layers];     // [4304]
    Weight fc2[VisionConfig::layers];          // [1152, 4304]
    Tensor fc2_bias[VisionConfig::layers];     // [1152]
    // main merger (use_postshuffle_norm=False): norm on hidden
    Tensor merger_norm_w;                      // [1152]
    Tensor merger_norm_b;                      // [1152]
    Weight merger_fc1;                         // [4608, 4608]
    Tensor merger_fc1_bias;                    // [4608]
    Weight merger_fc2;                         // [4096, 4608]
    Tensor merger_fc2_bias;                    // [4096]
    // deepstack mergers (use_postshuffle_norm=True): norm on merger_hidden
    Tensor ds_norm_w[VisionConfig::deepstack_layers];   // [4608]
    Tensor ds_norm_b[VisionConfig::deepstack_layers];   // [4608]
    Weight ds_fc1[VisionConfig::deepstack_layers];      // [4608, 4608]
    Tensor ds_fc1_bias[VisionConfig::deepstack_layers]; // [4608]
    Weight ds_fc2[VisionConfig::deepstack_layers];      // [4096, 4608]
    Tensor ds_fc2_bias[VisionConfig::deepstack_layers]; // [4096]
};

// Pre-allocated activation workspace for one image encode (sized for P patches, V=P/4 tokens).
struct VisionWorkspace {
    hip_bfloat16* patch_in = nullptr;  // [1536, P] patch-embed input (feature-major)
    hip_bfloat16* x        = nullptr;  // [1152, P] residual stream
    hip_bfloat16* h        = nullptr;  // [1152, P] normalized
    hip_bfloat16* qkv      = nullptr;  // [3456, P]
    hip_bfloat16* attended = nullptr;  // [1152, P]
    hip_bfloat16* projected = nullptr; // [1152, P]
    hip_bfloat16* up       = nullptr;  // [4304, P]
    hip_bfloat16* down     = nullptr;  // [1152, P]
    hip_bfloat16* merged   = nullptr;  // [4608, V] merger input view
    hip_bfloat16* mhidden  = nullptr;  // [4608, V] merger fc1 out
    int* position_ids      = nullptr;  // [P, 2]
    int* cu_seqlens        = nullptr;  // [segments+1]
    int* pos_indices       = nullptr;  // [4, P]
    float* pos_weights     = nullptr;  // [4, P]
};

// Encode one image: patches FP32 [1152, P] -> merged [4096, V] + 3 deepstack [4096, V].
// `deepstack_out` is [3][4096, V] (row-major, one [4096, V] block each).
void vision_forward(const VisionWeights& w, const float* patches, int patches_count,
                    int segment_count, const int* position_ids_host, const int* cu_seqlens_host,
                    const int* pos_indices_host, const float* pos_weights_host,
                    hip_bfloat16* merged_out, hip_bfloat16* deepstack_out, VisionWorkspace& ws,
                    hipStream_t stream);

} // namespace ninfer::targets::qwen3_vl::detail
