#pragma once

// Host-side preprocessing of one v2v condition image: decode, smart-resize, bicubic-resize,
// patchify, and the 4-corner position-embed controls. Produces exactly the inputs
// EncoderProgram::embed_v2v consumes. Mirrors the chat path's vision preprocessing
// (qwen3 frontend processor + vision control) so the same image yields the same grid,
// patches, and position interpolation.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_vl::detail {

struct V2VImagePrep {
    std::vector<float> patches;    // [1152, P] patch-major: 3 * 2 * 16 * 16 floats per patch
    std::vector<int> position_ids; // [P, 2]: (y, x) patch grid coordinates
    std::vector<int> pos_indices;  // [P][4]: corner indices into the 48x48 pos-embed table
    std::vector<float> pos_weights;// [P][4]: corner weights (sum to 1 per patch)
    int grid_h = 0;                // patch rows (resized height / 16)
    int grid_w = 0;                // patch columns (resized width / 16)
    int patches_count = 0;         // P = grid_h * grid_w
    int tokens = 0;                // V = P / 4 (2x2 spatial merge)
};

// Decodes JPEG/PNG bytes and preprocesses them for the ViT. Throws std::invalid_argument
// for undecodable input or a grid that overflows the position table.
V2VImagePrep preprocess_v2v_image(const std::uint8_t* bytes, std::size_t byte_count);

} // namespace ninfer::targets::qwen3_vl::detail
