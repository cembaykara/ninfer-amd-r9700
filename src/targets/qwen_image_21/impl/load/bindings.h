#pragma once

// Bind the 297 DiT tensors from the .ninfer artifact to the runtime DitWeights struct.
#include "../dit_forward.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"

namespace ninfer::targets::qwen_image_21::detail {

// Phase 1 (plan): require every DiT tensor on the binder; returns the handles.
struct DitWeightHandles {
    artifact::ObjectHandle img_in;
    artifact::ObjectHandle txt_in_in_layer, txt_in_out_layer, txt_in_text_norm;
    artifact::ObjectHandle time_embed_linear_1, time_embed_linear_2, modulation;
    artifact::ObjectHandle to_q[DitConfig::layers];
    artifact::ObjectHandle to_k[DitConfig::layers];
    artifact::ObjectHandle to_v[DitConfig::layers];
    artifact::ObjectHandle to_out[DitConfig::layers];
    artifact::ObjectHandle norm_q[DitConfig::layers];
    artifact::ObjectHandle norm_k[DitConfig::layers];
    artifact::ObjectHandle mlp_gate[DitConfig::layers];
    artifact::ObjectHandle mlp_up[DitConfig::layers];
    artifact::ObjectHandle mlp_down[DitConfig::layers];
    artifact::ObjectHandle norm_out_linear, proj_out;
};

DitWeightHandles plan_dit_weights(artifact::Binder& binder);

// Phase 2 (bind): populate DitWeights from the materialized device data.
void bind_dit_weights(DitWeights& w, const DitWeightHandles& h,
                     const artifact::MaterializedArtifact& materialized);

} // namespace ninfer::targets::qwen_image_21::detail
