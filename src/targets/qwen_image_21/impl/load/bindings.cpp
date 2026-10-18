#include "bindings.h"

#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen_image_21::detail {
namespace {

using artifact::NumericFormat;
using artifact::StorageLayout;

artifact::ObjectHandle req(artifact::Binder& b, const std::string& name, int out, int in) {
    return b.require_tensor(name, NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                            {static_cast<std::uint64_t>(out), static_cast<std::uint64_t>(in)});
}

artifact::ObjectHandle req1d(artifact::Binder& b, const std::string& name, int size) {
    return b.require_tensor(name, NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                            {static_cast<std::uint64_t>(size)});
}

Weight make_weight(const void* data, int out, int in) {
    Weight w;
    w.payload = data;
    w.qtype   = QType::BF16_CTRL;
    w.shape[0] = out;
    w.shape[1] = in;
    w.ndim     = 2;
    w.n        = out;
    w.k        = in;
    return w;
}

} // namespace

DitWeightHandles plan_dit_weights(artifact::Binder& b) {
    DitWeightHandles h;
    h.img_in            = req(b, "dit/img_in", 4096, 64);
    h.txt_in_in_layer   = req(b, "dit/txt_in/in_layer", 4096, 4096);
    h.txt_in_out_layer  = req(b, "dit/txt_in/out_layer", 4096, 4096);
    h.txt_in_text_norm  = req1d(b, "dit/txt_in/text_norm", 4096);
    h.time_embed_linear_1 = req(b, "dit/time_embed/linear_1", 4096, 256);
    h.time_embed_linear_2 = req(b, "dit/time_embed/linear_2", 4096, 4096);
    h.modulation        = req(b, "dit/modulation", 16384, 4096);
    for (int l = 0; l < DitConfig::layers; l++) {
        const std::string p = "dit/layers/" + std::to_string(l) + "/";
        h.to_q[l]   = req(b, p + "attention/query", 4096, 4096);
        h.to_k[l]   = req(b, p + "attention/key", 4096, 4096);
        h.to_v[l]   = req(b, p + "attention/value", 4096, 4096);
        h.to_out[l] = req(b, p + "attention/output", 4096, 4096);
        h.norm_q[l] = req1d(b, p + "attention/query_norm", 128);
        h.norm_k[l] = req1d(b, p + "attention/key_norm", 128);
        h.mlp_gate[l] = req(b, p + "mlp/gate", 12288, 4096);
        h.mlp_up[l]   = req(b, p + "mlp/up", 12288, 4096);
        h.mlp_down[l] = req(b, p + "mlp/down", 4096, 12288);
    }
    h.norm_out_linear = req(b, "dit/norm_out", 4096, 4096);
    h.proj_out        = req(b, "dit/proj_out", 64, 4096);
    return h;
}

void bind_dit_weights(DitWeights& w, const DitWeightHandles& h,
                     const artifact::MaterializedArtifact& m) {
    w.img_in            = make_weight(m.device_data(h.img_in), 4096, 64);
    w.txt_in_in_layer   = make_weight(m.device_data(h.txt_in_in_layer), 4096, 4096);
    w.txt_in_out_layer  = make_weight(m.device_data(h.txt_in_out_layer), 4096, 4096);
    w.txt_in_text_norm  = Tensor(static_cast<void*>(m.device_data(h.txt_in_text_norm)), DType::BF16, {4096});
    w.time_embed_linear_1 = make_weight(m.device_data(h.time_embed_linear_1), 4096, 256);
    w.time_embed_linear_2 = make_weight(m.device_data(h.time_embed_linear_2), 4096, 4096);
    w.modulation        = make_weight(m.device_data(h.modulation), 16384, 4096);
    for (int l = 0; l < DitConfig::layers; l++) {
        w.to_q[l]   = make_weight(m.device_data(h.to_q[l]), 4096, 4096);
        w.to_k[l]   = make_weight(m.device_data(h.to_k[l]), 4096, 4096);
        w.to_v[l]   = make_weight(m.device_data(h.to_v[l]), 4096, 4096);
        w.to_out[l] = make_weight(m.device_data(h.to_out[l]), 4096, 4096);
        w.norm_q[l] = Tensor(static_cast<void*>(m.device_data(h.norm_q[l])), DType::BF16, {128});
        w.norm_k[l] = Tensor(static_cast<void*>(m.device_data(h.norm_k[l])), DType::BF16, {128});
        w.mlp_gate[l] = make_weight(m.device_data(h.mlp_gate[l]), 12288, 4096);
        w.mlp_up[l]   = make_weight(m.device_data(h.mlp_up[l]), 12288, 4096);
        w.mlp_down[l] = make_weight(m.device_data(h.mlp_down[l]), 4096, 12288);
    }
    w.norm_out_linear = make_weight(m.device_data(h.norm_out_linear), 4096, 4096);
    w.proj_out        = make_weight(m.device_data(h.proj_out), 64, 4096);
}

} // namespace ninfer::targets::qwen_image_21::detail
