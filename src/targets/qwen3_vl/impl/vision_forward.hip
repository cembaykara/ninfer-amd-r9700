// Qwen3-VL vision (ViT) encoder forward (wave32 gfx1201).
#include "vision_forward.h"

#include "core/device.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/vision_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include <hip/hip_bfloat16.h>

namespace ninfer::targets::qwen3_vl::detail {
using namespace ninfer::ops;

void vision_forward(const VisionWeights& w, const float* patches, int patches_count,
                    int segment_count, const int* position_ids_host, const int* cu_seqlens_host,
                    const int* pos_indices_host, const float* pos_weights_host,
                    hip_bfloat16* merged_out, hip_bfloat16* deepstack_out, VisionWorkspace& ws,
                    hipStream_t stream) {
    const int H   = VisionConfig::hidden;      // 1152
    const int P   = patches_count;
    const int V   = P / VisionConfig::merge_unit;
    const float eps = VisionConfig::eps;

    // copy host control tensors to device
    HIP_CHECK(hipMemcpyAsync(ws.position_ids, position_ids_host, sizeof(int) * P * 2,
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(ws.cu_seqlens, cu_seqlens_host,
                             sizeof(int) * (segment_count + 1), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(ws.pos_indices, pos_indices_host, sizeof(int) * 4 * P,
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(ws.pos_weights, pos_weights_host, sizeof(float) * 4 * P,
                             hipMemcpyHostToDevice, stream));

    // Debug: staged dumps of the residual stream (NINFER_VIT_STAGES=path).
    std::vector<char> stages_buf;
    std::vector<char> qkv_buf;  // block 0 qkv [3456, P] bf16
    std::vector<char> att_buf;  // block 0 attention out [1152, P] bf16
    std::vector<char> qk_buf;   // block 0 q,k after rope [1152, P] x2 bf16
    auto dump_stage = [&]() {
        if (std::getenv("NINFER_VIT_STAGES") == nullptr) return;
        HIP_CHECK(hipStreamSynchronize(stream));
        std::size_t off = stages_buf.size();
        stages_buf.resize(off + (std::size_t)P * H * 2);
        HIP_CHECK(hipMemcpy(stages_buf.data() + off, ws.x, (std::size_t)P * H * 2,
                            hipMemcpyDeviceToHost));
    };

    // 1. patches FP32 [1536, P] (feature-major) -> BF16, then patch embed
    // (linear [1152,1536] + bias). The input is wider than the hidden, so it gets its
    // own buffer (the linear op rejects an in-place output).
    Tensor patch_f32(const_cast<float*>(patches), DType::FP32, {1536, P});
    Tensor x_in(ws.patch_in, DType::BF16, {1536, P});
    cast_fp32_to_bf16(patch_f32, x_in, stream);
    Tensor x(ws.x, DType::BF16, {H, P});
    linear(x_in, w.patch_embed, x, stream);
    add_bias(w.patch_embed_bias, x, stream);
    dump_stage();  // after patch_embed + bias

    // 2. position embed (4-corner interpolation). table is [1152, 2304] (zero-copy view).
    Tensor pos_table(w.pos_embed.data, DType::BF16, {H, VisionConfig::pos_embed});
    Tensor pos_idx(ws.pos_indices, DType::I32, {4, P});
    Tensor pos_wt(ws.pos_weights, DType::FP32, {4, P});
    vision_pos_embed_add(pos_table, pos_idx, pos_wt, x, stream);
    dump_stage();  // after pos_embed

    // 3. the 27 blocks.
    Tensor qkv_t(ws.qkv, DType::BF16, {3 * H, P});
    Tensor h_t(ws.h, DType::BF16, {H, P});
    Tensor attended_t(ws.attended, DType::BF16, {H, P});
    Tensor projected_t(ws.projected, DType::BF16, {H, P});
    Tensor up_t(ws.up, DType::BF16, {VisionConfig::intermediate, P});
    Tensor down_t(ws.down, DType::BF16, {H, P});
    Tensor pos_ids(ws.position_ids, DType::I32, {P, 2});

    for (int l = 0; l < VisionConfig::layers; l++) {
        // attention sub-block
        layer_norm(x, w.norm1_w[l], w.norm1_b[l], eps, h_t, stream);
        if (l == 0 && std::getenv("NINFER_VIT_STAGES") != nullptr) {
            HIP_CHECK(hipStreamSynchronize(stream));
            std::vector<char> hbuf((std::size_t)H * P * 2);
            HIP_CHECK(hipMemcpy(hbuf.data(), h_t.data, (std::size_t)H * P * 2,
                                hipMemcpyDeviceToHost));
            (void)hbuf;  // kept for layout; written in the main dump below
        }
        linear(h_t, w.qkv[l], qkv_t, stream);
        add_bias(w.qkv_bias[l], qkv_t, stream);
        if (l == 0 && std::getenv("NINFER_VIT_STAGES") != nullptr) {
            HIP_CHECK(hipStreamSynchronize(stream));
            qkv_buf.resize((std::size_t)3 * H * P * 2);
            HIP_CHECK(hipMemcpy(qkv_buf.data(), qkv_t.data, (std::size_t)3 * H * P * 2,
                                hipMemcpyDeviceToHost));
        }
        // split qkv [3*H, P] into q/k/v planes [72, 16, P]. qkv_t is feature-major, so each
        // token row of 3*H features is contiguous: the plane base is the section offset
        // (0/H/2*H) and the token stride is the full row (3*H), not the plane width (H).
        Tensor q(reinterpret_cast<hip_bfloat16*>(qkv_t.data), DType::BF16,
                 {VisionConfig::head_dim, VisionConfig::heads, P});
        Tensor k(reinterpret_cast<hip_bfloat16*>(qkv_t.data) + H, DType::BF16,
                 {VisionConfig::head_dim, VisionConfig::heads, P});
        Tensor v(reinterpret_cast<hip_bfloat16*>(qkv_t.data) + 2 * H, DType::BF16,
                 {VisionConfig::head_dim, VisionConfig::heads, P});
        const std::int64_t qkv_token_stride =
            static_cast<std::int64_t>(3 * H) * static_cast<std::int64_t>(sizeof(hip_bfloat16));
        q.nb[2] = qkv_token_stride;
        k.nb[2] = qkv_token_stride;
        v.nb[2] = qkv_token_stride;
        rope(pos_ids, VisionConfig::rotary_dim, VisionConfig::rope_theta, q, k, stream);
        if (l == 0 && std::getenv("NINFER_VIT_STAGES") != nullptr) {
            HIP_CHECK(hipStreamSynchronize(stream));
            qk_buf.resize((std::size_t)2 * H * P * 2);
            HIP_CHECK(hipMemcpy(qk_buf.data(), q.data, (std::size_t)H * P * 2,
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(qk_buf.data() + (std::size_t)H * P * 2, k.data,
                                (std::size_t)H * P * 2, hipMemcpyDeviceToHost));
        }
        Tensor attended_heads(attended_t.data, DType::BF16,
                              {VisionConfig::head_dim, VisionConfig::heads, P});
        // single image: fixed segment length P, no workspace needed
        vision_attention(q, k, v, P, attended_heads, stream);
        if (l == 0 && std::getenv("NINFER_VIT_STAGES") != nullptr) {
            HIP_CHECK(hipStreamSynchronize(stream));
            att_buf.resize((std::size_t)H * P * 2);
            HIP_CHECK(hipMemcpy(att_buf.data(), attended_t.data, (std::size_t)H * P * 2,
                                hipMemcpyDeviceToHost));
        }
        linear(attended_t, w.proj[l], projected_t, stream);
        add_bias(w.proj_bias[l], projected_t, stream);
        residual_add(projected_t, x, stream);

        // mlp sub-block
        layer_norm(x, w.norm2_w[l], w.norm2_b[l], eps, h_t, stream);
        linear(h_t, w.fc1[l], up_t, stream);
        add_bias(w.fc1_bias[l], up_t, stream);
        gelu(up_t, GeluMode::Tanh, stream);
        linear(up_t, w.fc2[l], down_t, stream);
        add_bias(w.fc2_bias[l], down_t, stream);
        residual_add(down_t, x, stream);
        if (l == 0) { dump_stage(); }  // after block 0

        // deepstack capture at layers 8/16/24
        for (int d = 0; d < VisionConfig::deepstack_layers; d++) {
            if (l == VisionConfig::deepstack_index[d]) {
                // view [1152, P] -> [4608, V] (spatial merge), norm, fc1, gelu, fc2
                Tensor merged_view(x.data, DType::BF16, {VisionConfig::merger_hidden, V});
                Tensor ds_normed(ws.merged, DType::BF16, {VisionConfig::merger_hidden, V});
                layer_norm(merged_view, w.ds_norm_w[d], w.ds_norm_b[d], eps, ds_normed, stream);
                Tensor ds_h(ws.mhidden, DType::BF16, {VisionConfig::merger_hidden, V});
                linear(ds_normed, w.ds_fc1[d], ds_h, stream);
                add_bias(w.ds_fc1_bias[d], ds_h, stream);
                gelu(ds_h, GeluMode::Exact, stream);
                Tensor ds_out(deepstack_out + d * VisionConfig::out_hidden * V, DType::BF16,
                              {VisionConfig::out_hidden, V});
                linear(ds_h, w.ds_fc2[d], ds_out, stream);
                add_bias(w.ds_fc2_bias[d], ds_out, stream);
            }
        }
    }

    dump_stage();  // pre-merger (after block 26)
    // 4. main merger: norm [1152,P] -> view [4608,V] -> fc1 -> gelu -> fc2 -> [4096,V]
    Tensor normed(ws.h, DType::BF16, {H, P});
    layer_norm(x, w.merger_norm_w, w.merger_norm_b, eps, normed, stream);
    Tensor merged_in(normed.data, DType::BF16, {VisionConfig::merger_hidden, V});
    Tensor mhidden(ws.mhidden, DType::BF16, {VisionConfig::merger_hidden, V});
    linear(merged_in, w.merger_fc1, mhidden, stream);
    add_bias(w.merger_fc1_bias, mhidden, stream);
    gelu(mhidden, GeluMode::Exact, stream);
    Tensor out(merged_out, DType::BF16, {VisionConfig::out_hidden, V});
    linear(mhidden, w.merger_fc2, out, stream);
    add_bias(w.merger_fc2_bias, out, stream);
    if (std::getenv("NINFER_VIT_STAGES") != nullptr) {
        std::ofstream f(std::getenv("NINFER_VIT_STAGES"), std::ios::binary);
        f.write(reinterpret_cast<const char*>(&P), 4);
        f.write(stages_buf.data(), (std::streamsize)stages_buf.size());
        // pos controls: indices [P*4] i32, weights [P*4] f32
        std::vector<char> cbuf((std::size_t)P * 4 * (sizeof(int) + sizeof(float)));
        HIP_CHECK(hipMemcpy(cbuf.data(), ws.pos_indices, (std::size_t)P * 4 * sizeof(int),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(cbuf.data() + (std::size_t)P * 4 * sizeof(int), ws.pos_weights,
                            (std::size_t)P * 4 * sizeof(float), hipMemcpyDeviceToHost));
        f.write(cbuf.data(), (std::streamsize)cbuf.size());
        f.write(qkv_buf.data(), (std::streamsize)qkv_buf.size());
        f.write(att_buf.data(), (std::streamsize)att_buf.size());
        f.write(qk_buf.data(), (std::streamsize)qk_buf.size());
        // block 0 qkv weight [3456, 1152] bf16
        std::vector<char> wbuf((std::size_t)3 * H * H * 2);
        HIP_CHECK(hipMemcpy(wbuf.data(), w.qkv[0].qdata, (std::size_t)3 * H * H * 2,
                            hipMemcpyDeviceToHost));
        f.write(wbuf.data(), (std::streamsize)wbuf.size());
        // block 0 norm1 weight/bias [1152] each
        std::vector<char> nbuf((std::size_t)H * 2 * 2);
        HIP_CHECK(hipMemcpy(nbuf.data(), w.norm1_w[0].data, (std::size_t)H * 2,
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(nbuf.data() + (std::size_t)H * 2, w.norm1_b[0].data,
                            (std::size_t)H * 2, hipMemcpyDeviceToHost));
        f.write(nbuf.data(), (std::streamsize)nbuf.size());
        // patch_in [1536, P] bf16 (the device input), then patch_embed weight [1152, 1536] and bias [1152]
        std::vector<char> ibuf((std::size_t)1536 * P * 2);
        HIP_CHECK(hipMemcpy(ibuf.data(), ws.patch_in, (std::size_t)1536 * P * 2,
                            hipMemcpyDeviceToHost));
        f.write(ibuf.data(), (std::streamsize)ibuf.size());
        // patch_embed weight [1152, 1536] bf16, then its bias [1152]
        std::vector<char> wbuf2((std::size_t)H * 1536 * 2 + H * 2);
        HIP_CHECK(hipMemcpy(wbuf2.data(), w.patch_embed.qdata, (std::size_t)H * 1536 * 2,
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(wbuf2.data() + (std::size_t)H * 1536 * 2, w.patch_embed_bias.data,
                            (std::size_t)H * 2, hipMemcpyDeviceToHost));
        f.write(wbuf2.data(), (std::streamsize)wbuf2.size());
    }
}

} // namespace ninfer::targets::qwen3_vl::detail
