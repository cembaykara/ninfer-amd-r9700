#pragma once

// Bind the text-encoder (398) + vision ViT (351) tensors from the .ninfer artifact to the
// runtime weights.
#include "../encoder_forward.h"
#include "../vision_forward.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"

namespace ninfer::targets::qwen3_vl::detail {

struct EncoderWeightHandles {
    artifact::ObjectHandle embed_tokens;
    artifact::ObjectHandle q_proj[EncoderConfig::layers];
    artifact::ObjectHandle k_proj[EncoderConfig::layers];
    artifact::ObjectHandle v_proj[EncoderConfig::layers];
    artifact::ObjectHandle o_proj[EncoderConfig::layers];
    artifact::ObjectHandle q_norm[EncoderConfig::layers];
    artifact::ObjectHandle k_norm[EncoderConfig::layers];
    artifact::ObjectHandle gate_proj[EncoderConfig::layers];
    artifact::ObjectHandle up_proj[EncoderConfig::layers];
    artifact::ObjectHandle down_proj[EncoderConfig::layers];
    artifact::ObjectHandle input_layernorm[EncoderConfig::layers];
    artifact::ObjectHandle post_attention_layernorm[EncoderConfig::layers];
    artifact::ObjectHandle final_norm;
};

struct VisionWeightHandles {
    artifact::ObjectHandle patch_embed;
    artifact::ObjectHandle patch_embed_bias;
    artifact::ObjectHandle pos_embed;
    artifact::ObjectHandle norm1_w[VisionConfig::layers];
    artifact::ObjectHandle norm1_b[VisionConfig::layers];
    artifact::ObjectHandle qkv[VisionConfig::layers];
    artifact::ObjectHandle qkv_bias[VisionConfig::layers];
    artifact::ObjectHandle proj[VisionConfig::layers];
    artifact::ObjectHandle proj_bias[VisionConfig::layers];
    artifact::ObjectHandle norm2_w[VisionConfig::layers];
    artifact::ObjectHandle norm2_b[VisionConfig::layers];
    artifact::ObjectHandle fc1[VisionConfig::layers];
    artifact::ObjectHandle fc1_bias[VisionConfig::layers];
    artifact::ObjectHandle fc2[VisionConfig::layers];
    artifact::ObjectHandle fc2_bias[VisionConfig::layers];
    artifact::ObjectHandle merger_norm_w;
    artifact::ObjectHandle merger_norm_b;
    artifact::ObjectHandle merger_fc1;
    artifact::ObjectHandle merger_fc1_bias;
    artifact::ObjectHandle merger_fc2;
    artifact::ObjectHandle merger_fc2_bias;
    artifact::ObjectHandle ds_norm_w[VisionConfig::deepstack_layers];
    artifact::ObjectHandle ds_norm_b[VisionConfig::deepstack_layers];
    artifact::ObjectHandle ds_fc1[VisionConfig::deepstack_layers];
    artifact::ObjectHandle ds_fc1_bias[VisionConfig::deepstack_layers];
    artifact::ObjectHandle ds_fc2[VisionConfig::deepstack_layers];
    artifact::ObjectHandle ds_fc2_bias[VisionConfig::deepstack_layers];
};

EncoderWeightHandles plan_encoder_weights(artifact::Binder& binder);
void bind_encoder_weights(EncoderWeights& w, const EncoderWeightHandles& h,
                          const artifact::MaterializedArtifact& materialized);

VisionWeightHandles plan_vision_weights(artifact::Binder& binder);
void bind_vision_weights(VisionWeights& w, const VisionWeightHandles& h,
                         const artifact::MaterializedArtifact& materialized);

} // namespace ninfer::targets::qwen3_vl::detail
