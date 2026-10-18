#include "bindings.h"

#include <array>
#include <string>

namespace ninfer::targets::qwen_image_vae {
namespace {

using artifact::NumericFormat;
using artifact::StorageLayout;

constexpr int kMidDim = 1152;
constexpr int kUpOut[5] = {1152, 1152, 576, 288, 144};
constexpr int kUpIn[5]  = {1152, 1152, 1152, 576, 288};

// 4D conv weight [a, b, c, e]
artifact::ObjectHandle req4d(artifact::Binder& b, const std::string& name, int a, int c, int d, int e) {
    const std::array<std::uint64_t, 4> shape{static_cast<std::uint64_t>(a), static_cast<std::uint64_t>(c),
                                             static_cast<std::uint64_t>(d), static_cast<std::uint64_t>(e)};
    return b.require_tensor(name, NumericFormat::BF16, StorageLayout::ContiguousLeV1, shape);
}
// 1D bias/norm [n]
artifact::ObjectHandle req1d(artifact::Binder& b, const std::string& name, int n) {
    const std::array<std::uint64_t, 1> shape{static_cast<std::uint64_t>(n)};
    return b.require_tensor(name, NumericFormat::BF16, StorageLayout::ContiguousLeV1, shape);
}
// 4D norm [c, 1, 1, 1]
artifact::ObjectHandle reqnorm4(artifact::Binder& b, const std::string& name, int c) {
    const std::array<std::uint64_t, 4> shape{static_cast<std::uint64_t>(c), 1, 1, 1};
    return b.require_tensor(name, NumericFormat::BF16, StorageLayout::ContiguousLeV1, shape);
}
// 3D norm [c, 1, 1] (attention norm)
artifact::ObjectHandle reqnorm3(artifact::Binder& b, const std::string& name, int c) {
    const std::array<std::uint64_t, 3> shape{static_cast<std::uint64_t>(c), 1, 1};
    return b.require_tensor(name, NumericFormat::BF16, StorageLayout::ContiguousLeV1, shape);
}

// Require one resnet's tensors (with optional shortcut)
void req_resnet(artifact::Binder& b, VaeWeightHandles::ResnetH& h, const std::string& p,
                int cin, int cout, bool shortcut) {
    h.conv1 = req4d(b, p + "conv1", cout, cin, 3, 3);
    h.conv1_b = req1d(b, p + "conv1_b", cout);
    h.conv2 = req4d(b, p + "conv2", cout, cout, 3, 3);
    h.conv2_b = req1d(b, p + "conv2_b", cout);
    h.norm1 = reqnorm4(b, p + "norm1", cin);
    h.norm2 = reqnorm4(b, p + "norm2", cout);
    if (shortcut) {
        h.shortcut = req4d(b, p + "shortcut", cout, cin, 1, 1);
        h.shortcut_b = req1d(b, p + "shortcut_b", cout);
    }
}

} // namespace

VaeWeightHandles plan_vae_weights(artifact::Binder& b) {
    VaeWeightHandles h;
    h.conv_in = req4d(b, "vae/conv_in", kMidDim, 64, 3, 3);
    h.conv_in_b = req1d(b, "vae/conv_in_b", kMidDim);

    for (int i = 0; i < 5; ++i) {
        const int cin = kUpIn[i], cout = kUpOut[i];
        const std::string base = "vae/up_blocks/" + std::to_string(i) + "/";
        for (int j = 0; j < 3; ++j) {
            const int rcin = (j == 0) ? cin : cout;
            req_resnet(b, h.up_blocks[i].resnets[j], base + "resnets/" + std::to_string(j) + "/",
                       rcin, cout, /*shortcut=*/(j == 0 && cin != cout));
        }
        if (i < 4) {
            h.up_blocks[i].resample = req4d(b, base + "upsampler/resample", cout, cout, 3, 3);
            h.up_blocks[i].resample_b = req1d(b, base + "upsampler/resample_b", cout);
        }
        if (i < 3) {
            h.up_blocks[i].time_conv = req4d(b, base + "upsampler/time_conv", 2 * cout, cout, 1, 1);
            h.up_blocks[i].time_conv_b = req1d(b, base + "upsampler/time_conv_b", 2 * cout);
        }
    }

    for (int j = 0; j < 2; ++j)
        req_resnet(b, h.mid_resnets[j], "vae/mid_block/resnets/" + std::to_string(j) + "/", kMidDim, kMidDim, false);
    h.mid_attn_to_qkv = req4d(b, "vae/mid_block/attn/to_qkv", 3 * kMidDim, kMidDim, 1, 1);
    h.mid_attn_to_qkv_b = req1d(b, "vae/mid_block/attn/to_qkv_b", 3 * kMidDim);
    h.mid_attn_proj = req4d(b, "vae/mid_block/attn/proj", kMidDim, kMidDim, 1, 1);
    h.mid_attn_proj_b = req1d(b, "vae/mid_block/attn/proj_b", kMidDim);
    h.mid_attn_norm = reqnorm3(b, "vae/mid_block/attn/norm", kMidDim);

    h.conv_out = req4d(b, "vae/conv_out", 4, kUpOut[4], 3, 3);
    h.conv_out_b = req1d(b, "vae/conv_out_b", 4);
    h.norm_out = reqnorm4(b, "vae/norm_out", kUpOut[4]);
    return h;
}

void bind_vae_weights(VaeWeights& w, const VaeWeightHandles& h, const artifact::MaterializedArtifact& m) {
    auto pd = [&](artifact::ObjectHandle oh) -> const hip_bfloat16* {
        return reinterpret_cast<const hip_bfloat16*>(m.device_data(oh));
    };
    w.conv_in = pd(h.conv_in);
    w.conv_in_b = pd(h.conv_in_b);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) {
            auto& r = w.up_blocks[i].resnets[j];
            auto& rh = h.up_blocks[i].resnets[j];
            r.conv1 = pd(rh.conv1); r.conv1_b = pd(rh.conv1_b);
            r.conv2 = pd(rh.conv2); r.conv2_b = pd(rh.conv2_b);
            r.norm1 = pd(rh.norm1); r.norm2 = pd(rh.norm2);
            r.shortcut = pd(rh.shortcut); r.shortcut_b = pd(rh.shortcut_b);
        }
        w.up_blocks[i].upsampler.resample = pd(h.up_blocks[i].resample);
        w.up_blocks[i].upsampler.resample_b = pd(h.up_blocks[i].resample_b);
        w.up_blocks[i].upsampler.time_conv = pd(h.up_blocks[i].time_conv);
        w.up_blocks[i].upsampler.time_conv_b = pd(h.up_blocks[i].time_conv_b);
    }
    for (int j = 0; j < 2; ++j) {
        auto& r = w.mid_resnets[j];
        auto& rh = h.mid_resnets[j];
        r.conv1 = pd(rh.conv1); r.conv1_b = pd(rh.conv1_b);
        r.conv2 = pd(rh.conv2); r.conv2_b = pd(rh.conv2_b);
        r.norm1 = pd(rh.norm1); r.norm2 = pd(rh.norm2);
    }
    w.mid_attn_to_qkv = pd(h.mid_attn_to_qkv);
    w.mid_attn_to_qkv_b = pd(h.mid_attn_to_qkv_b);
    w.mid_attn_proj = pd(h.mid_attn_proj);
    w.mid_attn_proj_b = pd(h.mid_attn_proj_b);
    w.mid_attn_norm = pd(h.mid_attn_norm);
    w.conv_out = pd(h.conv_out);
    w.conv_out_b = pd(h.conv_out_b);
    w.norm_out = pd(h.norm_out);
}

} // namespace ninfer::targets::qwen_image_vae
