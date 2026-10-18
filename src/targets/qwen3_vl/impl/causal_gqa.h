#pragma once

// Causal grouped-query attention for the Qwen3-VL text encoder prefill (wave32 gfx1201).
// q, out: [batch, tokens, heads, head_dim] BF16. k, v: [batch, tokens, kv_heads, head_dim] BF16.
// Query head h reads KV head h/gqa_group. Query at position i attends only to keys j <= i.
// Simple (non-flash) two-pass kernel: correct for validation; flash-attention for production.
#include "core/tensor.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen3_vl::detail {

void causal_gqa(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out, hipStream_t stream);

} // namespace ninfer::targets::qwen3_vl::detail
