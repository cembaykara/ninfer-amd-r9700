#pragma once

#include "core/tensor.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace ninfer::targets::qwen_image_vae {
namespace detail {

struct VaeWeights {
    // conv_in: [1152, 64, 3, 3] + bias [1152]
    const hip_bfloat16* conv_in      = nullptr;
    const hip_bfloat16* conv_in_b    = nullptr;
    // up_blocks (5 blocks, channels [1152,1152,576,288,144])
    // Each block: 3 resnets (conv1, conv2, norm1, norm2, optional shortcut) + upsampler
    struct Resnet {
        const hip_bfloat16* conv1 = nullptr;
        const hip_bfloat16* conv1_b = nullptr;
        const hip_bfloat16* conv2 = nullptr;
        const hip_bfloat16* conv2_b = nullptr;
        const hip_bfloat16* norm1 = nullptr;
        const hip_bfloat16* norm2 = nullptr;
        const hip_bfloat16* shortcut = nullptr;
        const hip_bfloat16* shortcut_b = nullptr;
    };
    struct Upsampler {
        const hip_bfloat16* resample = nullptr;
        const hip_bfloat16* resample_b = nullptr;
        const hip_bfloat16* time_conv = nullptr;
        const hip_bfloat16* time_conv_b = nullptr;
    };
    struct UpBlock {
        Resnet resnets[3];
        Upsampler upsampler;
    };
    UpBlock up_blocks[5];
    // mid_block: 2 resnets + 1 attention
    Resnet mid_resnets[2];
    const hip_bfloat16* mid_attn_to_qkv = nullptr;
    const hip_bfloat16* mid_attn_to_qkv_b = nullptr;
    const hip_bfloat16* mid_attn_proj = nullptr;
    const hip_bfloat16* mid_attn_proj_b = nullptr;
    const hip_bfloat16* mid_attn_norm = nullptr;
    // conv_out: [4, 144, 3, 3] + bias [4]
    const hip_bfloat16* conv_out = nullptr;
    const hip_bfloat16* conv_out_b = nullptr;
    // norm_out: [144, 1, 1, 1]
    const hip_bfloat16* norm_out = nullptr;
    // 1x1 applied to the scaled latent before conv_in. [64, 64, 1, 1] + bias [64].
    const hip_bfloat16* post_quant = nullptr;
    const hip_bfloat16* post_quant_b = nullptr;

    // --- Encoder (2D, for img2img): image [4,H,W] -> latent [64,H/16,W/16] ---
    // conv_in: [96, 4, 3, 3] + bias [96]
    const hip_bfloat16* enc_conv_in      = nullptr;
    const hip_bfloat16* enc_conv_in_b    = nullptr;
    // down_blocks (5 blocks, channels [96,96,192,384,768,768]); blocks 0-3 downsample 2x
    struct DownBlock {
        Resnet resnets[2];
        const hip_bfloat16* resample = nullptr;     // [cout, cout, 3, 3] (zero-pad + stride 2)
        const hip_bfloat16* resample_b = nullptr;   // [cout]
        bool has_down = false;
    };
    DownBlock enc_down_blocks[5];
    // mid_block: 2 resnets + 1 attention (768)
    Resnet enc_mid_resnets[2];
    const hip_bfloat16* enc_mid_attn_to_qkv = nullptr;
    const hip_bfloat16* enc_mid_attn_to_qkv_b = nullptr;
    const hip_bfloat16* enc_mid_attn_proj = nullptr;
    const hip_bfloat16* enc_mid_attn_proj_b = nullptr;
    const hip_bfloat16* enc_mid_attn_norm = nullptr;
    // conv_out: [128, 768, 3, 3] + bias [128]
    const hip_bfloat16* enc_conv_out = nullptr;
    const hip_bfloat16* enc_conv_out_b = nullptr;
    // quant_conv: [128, 128, 1, 1] + bias [128]
    const hip_bfloat16* enc_quant_conv = nullptr;
    const hip_bfloat16* enc_quant_conv_b = nullptr;
    // norm_out: [768, 1, 1, 1]
    const hip_bfloat16* enc_norm_out = nullptr;
};

struct VaeWorkspace {
    hip_bfloat16* a = nullptr;
    hip_bfloat16* b = nullptr;
    hip_bfloat16* c = nullptr;
    hip_bfloat16* d = nullptr;
    hip_bfloat16* e = nullptr;
    hip_bfloat16* f = nullptr;
    hip_bfloat16* g = nullptr;
    hip_bfloat16* h = nullptr;
    hip_bfloat16* i = nullptr;
    hip_bfloat16* j = nullptr;
    hip_bfloat16* k = nullptr;
    hip_bfloat16* l = nullptr;
    hip_bfloat16* m = nullptr;
    hip_bfloat16* n = nullptr;
    hip_bfloat16* o = nullptr;
    hip_bfloat16* p = nullptr;
    hip_bfloat16* q = nullptr;
    hip_bfloat16* r = nullptr;
    hip_bfloat16* s = nullptr;
    hip_bfloat16* t = nullptr;
    hip_bfloat16* u = nullptr;
    hip_bfloat16* v = nullptr;
    hip_bfloat16* w = nullptr;
    hip_bfloat16* x = nullptr;
    hip_bfloat16* y = nullptr;
    hip_bfloat16* z = nullptr;
    bool ready = false;
};

// VAE decode: latent [1, 64, H/16, W/16] -> image [1, 4, H, W]
void vae_decode(const VaeWeights& weights, const Tensor& latent, Tensor& out_image,
                VaeWorkspace& ws, hipStream_t stream);

// VAE encode (2D, for img2img): image [4, H, W] in [-1,1] -> latent [64, H/16, W/16]
// (the mode of the posterior; the logvar half is dropped).
void vae_encode(const VaeWeights& weights, const Tensor& image, Tensor& out_latent,
                VaeWorkspace& ws, hipStream_t stream);

} // namespace detail
} // namespace ninfer::targets::qwen_image_vae
