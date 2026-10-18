#pragma once

#include <cstdint>

// Qwen3-VL text encoder (T2I) geometry. Confirmed from the checkpoint config.json
// (text_config). Standard Qwen3 text stack: GQA 4:1, head_dim 128, mRoPE theta 5e6
// (collapses to 1-D RoPE for text-only input). One-shot prefill, no KV cache.
namespace ninfer::targets::qwen3_vl::detail {

struct EncoderConfig {
    static constexpr int hidden      = 4096;
    static constexpr int layers      = 36;
    static constexpr int heads       = 32;
    static constexpr int kv_heads    = 8;
    static constexpr int head_dim    = 128;
    static constexpr int query_size  = heads * head_dim;    // 4096
    static constexpr int kv_size     = kv_heads * head_dim; // 1024
    static constexpr int gqa_group   = heads / kv_heads;    // 4
    static constexpr int intermediate = 12288;
    static constexpr int vocab       = 151936;
    static constexpr int rope_theta  = 5000000;
    static constexpr float eps       = 1.0e-6F;
};

} // namespace ninfer::targets::qwen3_vl::detail
