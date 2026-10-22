#include "targets/qwen3_vl/impl/vision_preprocess.h"

#include "media/decode/decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_vl::detail {
namespace {

constexpr int kPatch        = 16;
constexpr int kTemporal     = 2;
constexpr int kMerge        = 2;
constexpr int kFactor       = kPatch * kMerge;
constexpr int kPatchFeatures = 3 * kTemporal * kPatch * kPatch;  // 1152
constexpr int kPositionSide = 48;  // the pos-embed table is 48x48 (2304 rows)

// The Turbo checkpoint's processor_config.json image_processor.size: shortest_edge /
// longest_edge. The chat path reads the same fields from its checkpoint's
// preprocessor_config.json; this target's tokenizer already comes from the same
// processor dir, so the bounds are fixed here.
constexpr std::uint64_t kMinPixels = 65536ULL;
constexpr std::uint64_t kMaxPixels = 16777216ULL;

struct Size {
    int h = 0;
    int w = 0;
};

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::invalid_argument(std::string(label) + " overflow");
    }
    return a * b;
}

int round_even(double value) { return static_cast<int>(std::nearbyint(value)); }

// The Qwen2-VL grid sizing (min/max pixels, aspect <= 200), as in the chat path's
// processor.cpp.
Size smart_resize_image(int height, int width, std::uint64_t min_pixels, std::uint64_t max_pixels) {
    if (height <= 0 || width <= 0 || min_pixels == 0 || max_pixels < min_pixels) {
        throw std::invalid_argument("invalid image resize configuration");
    }
    if (static_cast<double>(std::max(height, width)) / std::min(height, width) > 200.0) {
        throw std::invalid_argument("image aspect ratio must be at most 200");
    }
    int h                    = round_even(static_cast<double>(height) / kFactor) * kFactor;
    int w                    = round_even(static_cast<double>(width) / kFactor) * kFactor;
    const std::uint64_t area = checked_mul(static_cast<std::uint64_t>(h),
                                           static_cast<std::uint64_t>(w), "image area");
    if (area > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(height) * width / max_pixels);
        h = std::max(kFactor, static_cast<int>(std::floor(height / beta / kFactor)) * kFactor);
        w = std::max(kFactor, static_cast<int>(std::floor(width / beta / kFactor)) * kFactor);
    } else if (area < min_pixels) {
        const double beta =
            std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(height) * width));
        h = static_cast<int>(std::ceil(height * beta / kFactor)) * kFactor;
        w = static_cast<int>(std::ceil(width * beta / kFactor)) * kFactor;
    }
    return {h, w};
}

double cubic(double x) {
    // Torchvision's antialiased bicubic path uses the Keys/Pillow coefficient.
    constexpr double a = -0.5;
    x                  = std::abs(x);
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
    if (x < 2.0) { return (((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a); }
    return 0.0;
}

struct Coefficients {
    std::vector<int> starts;
    std::vector<int> offsets;
    std::vector<float> weights;
};

Coefficients coefficients(int input, int output) {
    Coefficients out;
    out.starts.resize(static_cast<std::size_t>(output));
    out.offsets.resize(static_cast<std::size_t>(output) + 1);
    const double scale    = static_cast<double>(input) / output;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double support  = 2.0 * (scale >= 1.0 ? scale : 1.0);
    for (int dst = 0; dst < output; ++dst) {
        const double center = scale * (dst + 0.5);
        // NOLINTNEXTLINE(bugprone-incorrect-roundings): torchvision bounds truncate x + 0.5
        const int begin = std::max(static_cast<int>(center - support + 0.5), 0);
        // NOLINTNEXTLINE(bugprone-incorrect-roundings): torchvision bounds truncate x + 0.5
        const int size = std::min(static_cast<int>(center + support + 0.5), input) - begin;
        out.starts[static_cast<std::size_t>(dst)]  = begin;
        out.offsets[static_cast<std::size_t>(dst)] = static_cast<int>(out.weights.size());
        double sum                                 = 0.0;
        for (int j = 0; j < size; ++j) {
            const double weight = cubic((j + begin - center + 0.5) * invscale);
            out.weights.push_back(static_cast<float>(weight));
            sum += weight;
        }
        if (sum == 0.0) { throw std::runtime_error("bicubic resize produced zero weights"); }
        const int first = out.offsets[static_cast<std::size_t>(dst)];
        for (std::size_t i = static_cast<std::size_t>(first); i < out.weights.size(); ++i) {
            out.weights[i] = static_cast<float>(out.weights[i] / sum);
        }
    }
    out.offsets[static_cast<std::size_t>(output)] = static_cast<int>(out.weights.size());
    return out;
}

media::decode::Image resize_bicubic(const media::decode::Image& input, Size size) {
    if (input.width == size.w && input.height == size.h) { return input; }
    const Coefficients horizontal = coefficients(input.width, size.w);
    const Coefficients vertical   = coefficients(input.height, size.h);
    std::vector<std::uint8_t> temp(static_cast<std::size_t>(input.height) * size.w * 3);
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < size.w; ++x) {
            const int first = horizontal.offsets[static_cast<std::size_t>(x)];
            const int last  = horizontal.offsets[static_cast<std::size_t>(x) + 1];
            for (int c = 0; c < 3; ++c) {
                float value = 0.0f;
                for (int i = first; i < last; ++i) {
                    const int source =
                        std::clamp(horizontal.starts[static_cast<std::size_t>(x)] + (i - first), 0,
                                   input.width - 1);
                    value +=
                        horizontal.weights[static_cast<std::size_t>(i)] *
                        input.rgb[(static_cast<std::size_t>(y) * input.width + source) * 3 + c];
                }
                temp[(static_cast<std::size_t>(y) * size.w + x) * 3 + c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value), 0, 255));
            }
        }
    }

    media::decode::Image out;
    out.width  = size.w;
    out.height = size.h;
    out.rgb.resize(static_cast<std::size_t>(size.h) * size.w * 3);
    for (int y = 0; y < size.h; ++y) {
        const int first = vertical.offsets[static_cast<std::size_t>(y)];
        const int last  = vertical.offsets[static_cast<std::size_t>(y) + 1];
        for (int x = 0; x < size.w; ++x) {
            for (int c = 0; c < 3; ++c) {
                float value = 0.0f;
                for (int i = first; i < last; ++i) {
                    const int source =
                        std::clamp(vertical.starts[static_cast<std::size_t>(y)] + (i - first), 0,
                                   input.height - 1);
                    value += vertical.weights[static_cast<std::size_t>(i)] *
                             temp[(static_cast<std::size_t>(source) * size.w + x) * 3 + c];
                }
                out.rgb[(static_cast<std::size_t>(y) * size.w + x) * 3 + c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value), 0, 255));
            }
        }
    }
    return out;
}

float normalized(const media::decode::Image& image, int y, int x, int channel) {
    return static_cast<float>(
               image.rgb[(static_cast<std::size_t>(y) * image.width + x) * 3 + channel]) /
               127.5f -
           1.0f;
}

// One 1152-float patch: channel-major, then the two (duplicated) temporal frames, then the
// 16x16 pixel block. Same order as the chat path's append_patch.
void append_patch(const media::decode::Image& frame, int grid_y, int grid_x,
                  std::vector<float>& out) {
    for (int channel = 0; channel < 3; ++channel) {
        for (int temporal = 0; temporal < kTemporal; ++temporal) {
            (void)temporal;  // a still image duplicates its single frame
            for (int y = 0; y < kPatch; ++y) {
                for (int x = 0; x < kPatch; ++x) {
                    out.push_back(
                        normalized(frame, grid_y * kPatch + y, grid_x * kPatch + x, channel));
                }
            }
        }
    }
}

// The 4-corner position-embed interpolation, as in the chat path's vision control:
// map the patch grid onto the 48x48 table and bilinearly interpolate.
float coordinate(int index, int size) {
    return size <= 1 ? 0.0F
                     : static_cast<float>(index) * static_cast<float>(kPositionSide - 1) /
                           static_cast<float>(size - 1);
}

} // namespace

V2VImagePrep preprocess_v2v_image(const std::uint8_t* bytes, std::size_t byte_count) {
    const media::decode::Policy policy;
    media::decode::Image image =
        media::decode::decode_image({bytes, byte_count}, policy);
    const Size size = smart_resize_image(image.height, image.width, kMinPixels, kMaxPixels);
    const int gh = size.h / kPatch;
    const int gw = size.w / kPatch;
    const int P  = gh * gw;
    if (P <= 0 || P % (kMerge * kMerge) != 0) {
        throw std::invalid_argument("v2v image grid is not merge-aligned");
    }
    image = resize_bicubic(image, size);

    V2VImagePrep out;
    out.grid_h        = gh;
    out.grid_w        = gw;
    out.patches_count = P;
    out.tokens        = P / (kMerge * kMerge);
    out.patches.reserve(static_cast<std::size_t>(P) * kPatchFeatures);
    for (int block_y = 0; block_y < gh / kMerge; ++block_y) {
        for (int block_x = 0; block_x < gw / kMerge; ++block_x) {
            for (int merge_y = 0; merge_y < kMerge; ++merge_y) {
                for (int merge_x = 0; merge_x < kMerge; ++merge_x) {
                    append_patch(image, block_y * kMerge + merge_y, block_x * kMerge + merge_x,
                                 out.patches);
                }
            }
        }
    }

    out.position_ids.reserve(static_cast<std::size_t>(P) * 2);
    out.pos_indices.reserve(static_cast<std::size_t>(P) * 4);
    out.pos_weights.reserve(static_cast<std::size_t>(P) * 4);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            out.position_ids.push_back(y);
            out.position_ids.push_back(x);
            const float yf = coordinate(y, gh);
            const float xf = coordinate(x, gw);
            const int y0   = static_cast<int>(yf);
            const int x0   = static_cast<int>(xf);
            const int y1   = std::min(y0 + 1, kPositionSide - 1);
            const int x1   = std::min(x0 + 1, kPositionSide - 1);
            const float wy = yf - static_cast<float>(y0);
            const float wx = xf - static_cast<float>(x0);
            out.pos_indices.insert(out.pos_indices.end(),
                                   {y0 * kPositionSide + x0, y0 * kPositionSide + x1,
                                    y1 * kPositionSide + x0, y1 * kPositionSide + x1});
            out.pos_weights.insert(out.pos_weights.end(),
                                   {(1.0F - wy) * (1.0F - wx), (1.0F - wy) * wx,
                                    wy * (1.0F - wx), wy * wx});
        }
    }
    return out;
}

} // namespace ninfer::targets::qwen3_vl::detail
