#pragma once

// Bind the 134 VAE decoder tensors from the .ninfer artifact to the runtime VaeWeights struct.
#include "../vae_forward.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"

namespace ninfer::targets::qwen_image_vae {

struct VaeWeightHandles {
    artifact::ObjectHandle conv_in, conv_in_b;
    struct ResnetH {
        artifact::ObjectHandle conv1, conv1_b, conv2, conv2_b, norm1, norm2, shortcut, shortcut_b;
        bool has_shortcut = false;
    };
    struct UpBlockH {
        ResnetH resnets[3];
        artifact::ObjectHandle resample, resample_b, time_conv, time_conv_b;
    };
    UpBlockH up_blocks[5];
    ResnetH mid_resnets[2];
    artifact::ObjectHandle mid_attn_to_qkv, mid_attn_to_qkv_b, mid_attn_proj, mid_attn_proj_b, mid_attn_norm;
    artifact::ObjectHandle conv_out, conv_out_b, norm_out;
    artifact::ObjectHandle post_quant, post_quant_b;

    // --- Encoder (2D, for img2img) ---
    artifact::ObjectHandle enc_conv_in, enc_conv_in_b;
    struct DownBlockH {
        ResnetH resnets[2];
        artifact::ObjectHandle resample, resample_b;
        bool has_down = false;
    };
    DownBlockH enc_down_blocks[5];
    ResnetH enc_mid_resnets[2];
    artifact::ObjectHandle enc_mid_attn_to_qkv, enc_mid_attn_to_qkv_b, enc_mid_attn_proj, enc_mid_attn_proj_b, enc_mid_attn_norm;
    artifact::ObjectHandle enc_conv_out, enc_conv_out_b;
    artifact::ObjectHandle enc_quant_conv, enc_quant_conv_b;
    artifact::ObjectHandle enc_norm_out;
};

VaeWeightHandles plan_vae_weights(artifact::Binder& binder);
void bind_vae_weights(detail::VaeWeights& w, const VaeWeightHandles& h, const artifact::MaterializedArtifact& m);

} // namespace ninfer::targets::qwen_image_vae
