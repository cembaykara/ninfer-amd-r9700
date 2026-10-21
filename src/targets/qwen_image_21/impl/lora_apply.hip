#include "lora_apply.h"

#include "lora_math.h"
#include "lora_safetensors.h"

#include "artifact/reader.h"
#include "core/device.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen_image_21::detail {
namespace {

__global__ void add_lora_kernel(std::uint16_t* weight, int rows, int cols, int rank, const std::uint16_t* b,
                                const std::uint16_t* a, float scale) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int n   = rows * cols;
    if (idx >= n) { return; }
    const int row = idx / cols;
    const int col = idx - row * cols;
    weight[idx]   = ninfer_lora_element(weight[idx], b + static_cast<std::size_t>(row) * rank, a, cols, rank, col,
                                      scale);
}

const Weight& weight_for(const DitWeights& weights, const LoraPatch& patch) {
    switch (patch.slot) {
    case LoraSlot::Query:
        return weights.to_q[patch.layer];
    case LoraSlot::Key:
        return weights.to_k[patch.layer];
    case LoraSlot::Value:
        return weights.to_v[patch.layer];
    case LoraSlot::Output:
        return weights.to_out[patch.layer];
    case LoraSlot::Gate:
        return weights.mlp_gate[patch.layer];
    case LoraSlot::Up:
        return weights.mlp_up[patch.layer];
    case LoraSlot::Down:
        return weights.mlp_down[patch.layer];
    }
    throw std::runtime_error("image_denoise: unknown LoRA slot");
}

struct Touched {
    void* device = nullptr;
    std::string name;
    std::uint64_t bytes = 0;
};

void restore_weights(const std::filesystem::path& artifact, const std::vector<Touched>& touched, hipStream_t stream) {
    if (touched.empty()) { return; }
    // Read-only. The bytes copied back are the ones load put on the device, so the base file
    // stays the base file and the next denoise without a LoRA sees those weights again.
    const artifact::Reader reader(artifact);
    for (const Touched& tensor : touched) {
        const artifact::PayloadSpan span = reader.payload(tensor.name);
        if (span.data.size() != tensor.bytes) {
            throw std::runtime_error("image_denoise: LoRA restore size mismatch for " + tensor.name);
        }
        HIP_CHECK(hipMemcpyAsync(tensor.device, span.data.data(), tensor.bytes, hipMemcpyHostToDevice, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

} // namespace

void denoise_with_lora(const DitWeights& weights, const std::filesystem::path& artifact,
                       const std::vector<LoraRef>& loras, hipStream_t stream,
                       const std::function<void()>& denoise) {
    if (artifact.empty()) { throw std::runtime_error("image_denoise: loaded artifact path is empty"); }

    std::vector<Touched> touched;
    std::vector<void*> temps;
    const auto free_temps = [&]() {
        for (void* ptr : temps) { (void)hipFree(ptr); }
        temps.clear();
    };
    try {
        for (const LoraRef& lora : loras) {
            if (!std::isfinite(lora.scale)) {
                throw std::invalid_argument("image_denoise: lora scale must be finite");
            }
            const std::vector<LoraPatch> patches = read_lora_patches(lora.path);
            for (const LoraPatch& patch : patches) {
                const Weight& weight = weight_for(weights, patch);
                const auto bytes = static_cast<std::uint64_t>(patch.rows) * static_cast<std::uint64_t>(patch.cols) * 2;
                if (weight.payload == nullptr || weight.n != patch.rows || weight.k != patch.cols ||
                    weight.payload_bytes != bytes) {
                    throw std::invalid_argument("image_denoise: lora tensor does not match " + patch.tensor_name);
                }
                void* a_dev = nullptr;
                void* b_dev = nullptr;
                HIP_CHECK(hipMalloc(&a_dev, patch.a.size() * sizeof(std::uint16_t)));
                temps.push_back(a_dev);
                HIP_CHECK(hipMalloc(&b_dev, patch.b.size() * sizeof(std::uint16_t)));
                temps.push_back(b_dev);
                HIP_CHECK(hipMemcpyAsync(a_dev, patch.a.data(), patch.a.size() * sizeof(std::uint16_t),
                                         hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(b_dev, patch.b.data(), patch.b.size() * sizeof(std::uint16_t),
                                         hipMemcpyHostToDevice, stream));
                const int n       = patch.rows * patch.cols;
                const int threads = 256;
                const int blocks  = (n + threads - 1) / threads;
                auto* device_weight = static_cast<std::uint16_t*>(const_cast<void*>(weight.payload));
                add_lora_kernel<<<blocks, threads, 0, stream>>>(device_weight, patch.rows, patch.cols, patch.rank,
                                                                static_cast<const std::uint16_t*>(b_dev),
                                                                static_cast<const std::uint16_t*>(a_dev), lora.scale);
                touched.push_back(Touched{device_weight, patch.tensor_name, bytes});
            }
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        free_temps();
    } catch (...) {
        (void)hipStreamSynchronize(stream);
        free_temps();
        restore_weights(artifact, touched, stream);
        throw;
    }

    try {
        denoise();
    } catch (...) {
        restore_weights(artifact, touched, stream);
        throw;
    }
    restore_weights(artifact, touched, stream);
}

} // namespace ninfer::targets::qwen_image_21::detail
