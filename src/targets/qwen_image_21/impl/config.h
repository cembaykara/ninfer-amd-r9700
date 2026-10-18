#pragma once

#include <cstdint>

// Qwen-Image-2.1-Turbo DiT (transformer) geometry. Confirmed from the diffusers
// source (transformer_qwenimage21.py) and the checkpoint config.json. Single-stream:
// one shared modulation, block-causal attention (text causal, image bidirectional).
namespace ninfer::targets::qwen_image_21::detail {

struct DitConfig {
    static constexpr int hidden         = 4096;
    static constexpr int layers         = 32;
    static constexpr int heads          = 32;
    static constexpr int head_dim       = 128;
    static constexpr int query_size     = heads * head_dim;   // 4096
    static constexpr int mlp_ratio      = 3;
    static constexpr int intermediate   = mlp_ratio * hidden; // 12288
    static constexpr int in_channels    = 64;
    static constexpr int out_channels   = 64;
    static constexpr int context_in_dim = 4096;
    static constexpr int patch_size     = 1;                  // no spatial patchification
    static constexpr int rope_theta     = 10000;
    static constexpr int rope_axes[3]   = {16, 56, 56};       // frame, height, width
    static constexpr int rope_complex   = (rope_axes[0] + rope_axes[1] + rope_axes[2]) / 2; // 64
    static constexpr float eps          = 1.0e-6F;
    static constexpr int time_embed_dim = 256;                // sinusoidal width
    static constexpr int modulation_out = 4 * hidden;         // 16384

    // The joint sequence is [text (M), target-image (N)]; only the image rows of the
    // final proj_out are the velocity. For T2I the block-causal mask reduces to full
    // bidirectional attention over the joint sequence (text outputs discarded).
    static constexpr int modulation_scale1 = 0;
    static constexpr int modulation_gate1  = hidden;
    static constexpr int modulation_scale2 = 2 * hidden;
    static constexpr int modulation_gate2  = 3 * hidden;
};

} // namespace ninfer::targets::qwen_image_21::detail
