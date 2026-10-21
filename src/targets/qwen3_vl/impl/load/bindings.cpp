#include "bindings.h"

#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_vl::detail {
namespace {

using artifact::NumericFormat;
using artifact::StorageLayout;

artifact::ObjectHandle req(artifact::Binder& b, const std::string& name, int out, int in) {
    const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(out), static_cast<std::uint64_t>(in)};
    const auto h_ = b.require_tensor(name, NumericFormat::BF16, StorageLayout::ContiguousLeV1, shape);
    b.materialize_on_device(h_);
    return h_;
}

artifact::ObjectHandle req1d(artifact::Binder& b, const std::string& name, int size) {
    const std::array<std::uint64_t, 1> shape{static_cast<std::uint64_t>(size)};
    const auto h_ = b.require_tensor(name, NumericFormat::BF16, StorageLayout::ContiguousLeV1, shape);
    b.materialize_on_device(h_);
    return h_;
}

// 5D tensor (the patch-embed conv weight [1152, 3, 2, 16, 16]).
artifact::ObjectHandle req5d(artifact::Binder& b, const std::string& name,
                             int a, int c, int t, int ph, int pw) {
    const std::array<std::uint64_t, 5> shape{static_cast<std::uint64_t>(a),
                                             static_cast<std::uint64_t>(c),
                                             static_cast<std::uint64_t>(t),
                                             static_cast<std::uint64_t>(ph),
                                             static_cast<std::uint64_t>(pw)};
    const auto h_ = b.require_tensor(name, NumericFormat::BF16, StorageLayout::ContiguousLeV1, shape);
    b.materialize_on_device(h_);
    return h_;
}

Weight make_weight(const void* data, int out, int in) {
    // Same fields as artifact::contiguous_weight: ops::linear and ops::embedding read qdata
    // and check the byte counts, layout and padded shape.
    Weight w;
    w.payload         = data;
    w.qdata           = data;
    w.payload_bytes   = static_cast<std::uint64_t>(out) * in * 2;
    w.qdata_bytes     = w.payload_bytes;
    w.qtype           = QType::BF16_CTRL;
    w.layout          = QuantLayout::Contiguous;
    w.shape[0]        = out;
    w.shape[1]        = in;
    w.padded_shape[0] = out;
    w.padded_shape[1] = in;
    w.ndim            = 2;
    w.n               = out;
    w.k               = in;
    return w;
}

Tensor make_tensor(const void* data, int size) {
    return Tensor(static_cast<void*>(const_cast<void*>(data)), DType::BF16, {size});
}

} // namespace

EncoderWeightHandles plan_encoder_weights(artifact::Binder& b) {
    EncoderWeightHandles h;
    h.embed_tokens = req(b, "encoder/embed_tokens", 151936, 4096);
    for (int l = 0; l < EncoderConfig::layers; l++) {
        const std::string p = "encoder/layers/" + std::to_string(l) + "/";
        h.q_proj[l]  = req(b, p + "attention/query", 4096, 4096);
        h.k_proj[l]  = req(b, p + "attention/key", 1024, 4096);
        h.v_proj[l]  = req(b, p + "attention/value", 1024, 4096);
        h.o_proj[l]  = req(b, p + "attention/output", 4096, 4096);
        h.q_norm[l]  = req1d(b, p + "attention/query_norm", 128);
        h.k_norm[l]  = req1d(b, p + "attention/key_norm", 128);
        h.gate_proj[l] = req(b, p + "mlp/gate", 12288, 4096);
        h.up_proj[l]   = req(b, p + "mlp/up", 12288, 4096);
        h.down_proj[l] = req(b, p + "mlp/down", 4096, 12288);
        h.input_layernorm[l] = req1d(b, p + "input_layernorm", 4096);
        h.post_attention_layernorm[l] = req1d(b, p + "post_attention_layernorm", 4096);
    }
    h.final_norm = req1d(b, "encoder/final_norm", 4096);
    return h;
}

void bind_encoder_weights(EncoderWeights& w, const EncoderWeightHandles& h,
                          const artifact::MaterializedArtifact& m) {
    w.embed_tokens = make_weight(m.device_data(h.embed_tokens), 151936, 4096);
    for (int l = 0; l < EncoderConfig::layers; l++) {
        w.q_proj[l]  = make_weight(m.device_data(h.q_proj[l]), 4096, 4096);
        w.k_proj[l]  = make_weight(m.device_data(h.k_proj[l]), 1024, 4096);
        w.v_proj[l]  = make_weight(m.device_data(h.v_proj[l]), 1024, 4096);
        w.o_proj[l]  = make_weight(m.device_data(h.o_proj[l]), 4096, 4096);
        w.q_norm[l]  = make_tensor(m.device_data(h.q_norm[l]), 128);
        w.k_norm[l]  = make_tensor(m.device_data(h.k_norm[l]), 128);
        w.gate_proj[l] = make_weight(m.device_data(h.gate_proj[l]), 12288, 4096);
        w.up_proj[l]   = make_weight(m.device_data(h.up_proj[l]), 12288, 4096);
        w.down_proj[l] = make_weight(m.device_data(h.down_proj[l]), 4096, 12288);
        w.input_layernorm[l] = make_tensor(m.device_data(h.input_layernorm[l]), 4096);
        w.post_attention_layernorm[l] = make_tensor(m.device_data(h.post_attention_layernorm[l]), 4096);
    }
    w.final_norm = make_tensor(m.device_data(h.final_norm), 4096);
}

VisionWeightHandles plan_vision_weights(artifact::Binder& b) {
    VisionWeightHandles h;
    h.patch_embed = req5d(b, "visual/patch_embed", 1152, 3, 2, 16, 16);
    h.patch_embed_bias = req1d(b, "visual/patch_embed_b", 1152);
    h.pos_embed = req(b, "visual/pos_embed", 2304, 1152);
    for (int l = 0; l < VisionConfig::layers; l++) {
        const std::string p = "visual/blocks/" + std::to_string(l) + "/";
        h.qkv[l]        = req(b, p + "attn/qkv", 3456, 1152);
        h.qkv_bias[l]   = req1d(b, p + "attn/qkv_b", 3456);
        h.proj[l]       = req(b, p + "attn/proj", 1152, 1152);
        h.proj_bias[l]  = req1d(b, p + "attn/proj_b", 1152);
        h.fc1[l]        = req(b, p + "mlp/fc1", 4304, 1152);
        h.fc1_bias[l]   = req1d(b, p + "mlp/fc1_b", 4304);
        h.fc2[l]        = req(b, p + "mlp/fc2", 1152, 4304);
        h.fc2_bias[l]   = req1d(b, p + "mlp/fc2_b", 1152);
        h.norm1_w[l]    = req1d(b, p + "norm1", 1152);
        h.norm1_b[l]    = req1d(b, p + "norm1_b", 1152);
        h.norm2_w[l]    = req1d(b, p + "norm2", 1152);
        h.norm2_b[l]    = req1d(b, p + "norm2_b", 1152);
    }
    h.merger_norm_w = req1d(b, "visual/merger/norm", 1152);
    h.merger_norm_b = req1d(b, "visual/merger/norm_b", 1152);
    h.merger_fc1 = req(b, "visual/merger/fc1", 4608, 4608);
    h.merger_fc1_bias = req1d(b, "visual/merger/fc1_b", 4608);
    h.merger_fc2 = req(b, "visual/merger/fc2", 4096, 4608);
    h.merger_fc2_bias = req1d(b, "visual/merger/fc2_b", 4096);
    for (int d = 0; d < VisionConfig::deepstack_layers; d++) {
        const std::string p = "visual/deepstack/" + std::to_string(d) + "/";
        h.ds_norm_w[d]    = req1d(b, p + "norm", 4608);
        h.ds_norm_b[d]    = req1d(b, p + "norm_b", 4608);
        h.ds_fc1[d]       = req(b, p + "fc1", 4608, 4608);
        h.ds_fc1_bias[d]  = req1d(b, p + "fc1_b", 4608);
        h.ds_fc2[d]       = req(b, p + "fc2", 4096, 4608);
        h.ds_fc2_bias[d]  = req1d(b, p + "fc2_b", 4096);
    }
    return h;
}

void bind_vision_weights(VisionWeights& w, const VisionWeightHandles& h,
                         const artifact::MaterializedArtifact& m) {
    // patch_embed is stored 5D [1152,3,2,16,16]; the conv with kernel==stride is a plain
    // [1152,1152] linear over the flattened patch, so bind it as a 2D weight (same payload).
    w.patch_embed = make_weight(m.device_data(h.patch_embed), 1152, 1152);
    w.patch_embed_bias = make_tensor(m.device_data(h.patch_embed_bias), 1152);
    w.pos_embed = Tensor(static_cast<void*>(const_cast<void*>(m.device_data(h.pos_embed))),
                         DType::BF16, {2304, 1152});
    for (int l = 0; l < VisionConfig::layers; l++) {
        w.qkv[l]        = make_weight(m.device_data(h.qkv[l]), 3456, 1152);
        w.qkv_bias[l]   = make_tensor(m.device_data(h.qkv_bias[l]), 3456);
        w.proj[l]       = make_weight(m.device_data(h.proj[l]), 1152, 1152);
        w.proj_bias[l]  = make_tensor(m.device_data(h.proj_bias[l]), 1152);
        w.fc1[l]        = make_weight(m.device_data(h.fc1[l]), 4304, 1152);
        w.fc1_bias[l]   = make_tensor(m.device_data(h.fc1_bias[l]), 4304);
        w.fc2[l]        = make_weight(m.device_data(h.fc2[l]), 1152, 4304);
        w.fc2_bias[l]   = make_tensor(m.device_data(h.fc2_bias[l]), 1152);
        w.norm1_w[l]    = make_tensor(m.device_data(h.norm1_w[l]), 1152);
        w.norm1_b[l]    = make_tensor(m.device_data(h.norm1_b[l]), 1152);
        w.norm2_w[l]    = make_tensor(m.device_data(h.norm2_w[l]), 1152);
        w.norm2_b[l]    = make_tensor(m.device_data(h.norm2_b[l]), 1152);
    }
    w.merger_norm_w = make_tensor(m.device_data(h.merger_norm_w), 1152);
    w.merger_norm_b = make_tensor(m.device_data(h.merger_norm_b), 1152);
    w.merger_fc1 = make_weight(m.device_data(h.merger_fc1), 4608, 4608);
    w.merger_fc1_bias = make_tensor(m.device_data(h.merger_fc1_bias), 4608);
    w.merger_fc2 = make_weight(m.device_data(h.merger_fc2), 4096, 4608);
    w.merger_fc2_bias = make_tensor(m.device_data(h.merger_fc2_bias), 4096);
    for (int d = 0; d < VisionConfig::deepstack_layers; d++) {
        w.ds_norm_w[d]    = make_tensor(m.device_data(h.ds_norm_w[d]), 4608);
        w.ds_norm_b[d]    = make_tensor(m.device_data(h.ds_norm_b[d]), 4608);
        w.ds_fc1[d]       = make_weight(m.device_data(h.ds_fc1[d]), 4608, 4608);
        w.ds_fc1_bias[d]  = make_tensor(m.device_data(h.ds_fc1_bias[d]), 4608);
        w.ds_fc2[d]       = make_weight(m.device_data(h.ds_fc2[d]), 4096, 4608);
        w.ds_fc2_bias[d]  = make_tensor(m.device_data(h.ds_fc2_bias[d]), 4096);
    }
}

} // namespace ninfer::targets::qwen3_vl::detail
