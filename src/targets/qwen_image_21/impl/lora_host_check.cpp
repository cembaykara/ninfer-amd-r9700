// Host check for the denoise LoRA math and safetensors reader.
//
//   c++ -std=c++20 -O2 -I src/targets/qwen_image_21/impl \
//     src/targets/qwen_image_21/impl/lora_safetensors.cpp \
//     src/targets/qwen_image_21/impl/lora_host_check.cpp -o /tmp/lora_host_check
//   /tmp/lora_host_check /path/to/lora.safetensors
//
// Checks scale * (B @ A) and that reading a LoRA does not change the file.
// It does not load a DiT, start a server, or write a merged .ninfer file.
#include "lora_math.h"
#include "lora_safetensors.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ninfer::targets::qwen_image_21::detail::LoraPatch;
using ninfer::targets::qwen_image_21::detail::LoraSlot;
using ninfer::targets::qwen_image_21::detail::read_lora_patches;

namespace {

int fail(const std::string& message) {
    std::cerr << message << "\n";
    return 1;
}

std::uint64_t fnv(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::uint64_t hash = 14695981039346656037ull;
    char buf[1 << 16];
    while (in) {
        in.read(buf, sizeof(buf));
        const auto n = static_cast<std::size_t>(in.gcount());
        for (std::size_t i = 0; i < n; ++i) {
            hash ^= static_cast<unsigned char>(buf[i]);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

void check_hand_math() {
    constexpr int kRank = 2;
    constexpr int kCols = 3;
    constexpr int kRows = 2;
    const std::uint16_t one = ninfer_bf16_from_f32(1.F);
    const std::uint16_t two = ninfer_bf16_from_f32(2.F);
    // A is (2, 3): [[1, 0, 2], [0, 1, 0]]. B is (2, 2): [[1, 0], [0, 2]].
    const std::uint16_t a[] = {one, 0, two, 0, one, 0};
    const std::uint16_t b[] = {one, 0, 0, two};
    std::uint16_t weight[kRows * kCols] = {};
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            weight[row * kCols + col] =
                ninfer_lora_element(0, b + row * kRank, a, kCols, kRank, col, 1.F);
        }
    }
    const std::uint16_t want_one[] = {one, 0, two, 0, two, 0};
    for (int i = 0; i < kRows * kCols; ++i) {
        if (weight[i] != want_one[i]) { throw std::runtime_error("scale 1 hand math"); }
    }
    const std::uint16_t half = ninfer_bf16_from_f32(0.5F);
    std::uint16_t half_w[kRows * kCols] = {};
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            half_w[row * kCols + col] =
                ninfer_lora_element(0, b + row * kRank, a, kCols, kRank, col, 0.5F);
        }
    }
    const std::uint16_t want_half[] = {half, 0, one, 0, one, 0};
    for (int i = 0; i < kRows * kCols; ++i) {
        if (half_w[i] != want_half[i]) { throw std::runtime_error("scale 0.5 hand math"); }
    }
}

const LoraPatch& find_patch(const std::vector<LoraPatch>& patches, int layer, LoraSlot slot) {
    for (const LoraPatch& patch : patches) {
        if (patch.layer == layer && patch.slot == slot) { return patch; }
    }
    throw std::runtime_error("missing patch");
}

void print_element(const char* name, const LoraPatch& patch, int row, int col, float scale,
                   std::uint16_t weight) {
    const std::uint16_t got = ninfer_lora_element(weight, patch.b.data() + static_cast<std::size_t>(row) * patch.rank,
                                                  patch.a.data(), patch.cols, patch.rank, col, scale);
    std::cout << "cross " << name << " " << row << " " << col << " " << scale << " " << std::hex << weight << " "
              << got << std::dec << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        check_hand_math();
        if (argc < 2) { return fail("usage: lora_host_check <lora.safetensors>"); }
        const fs::path lora = argv[1];
        const auto before = fnv(lora);
        const std::uintmax_t bytes_before = fs::file_size(lora);
        const std::vector<LoraPatch> patches = read_lora_patches(lora);
        if (fnv(lora) != before || fs::file_size(lora) != bytes_before) {
            return fail("reading the LoRA changed the file");
        }
        if (patches.size() != 224) { return fail("expected 224 patches, got " + std::to_string(patches.size())); }
        const LoraPatch& query = find_patch(patches, 0, LoraSlot::Query);
        const LoraPatch& gate  = find_patch(patches, 0, LoraSlot::Gate);
        const LoraPatch& up    = find_patch(patches, 0, LoraSlot::Up);
        const LoraPatch& down  = find_patch(patches, 0, LoraSlot::Down);
        if (query.tensor_name != "dit/layers/0/attention/query" || query.rows != 4096 || query.cols != 4096 ||
            query.rank != 16) {
            return fail("query patch shape");
        }
        if (gate.tensor_name != "dit/layers/0/mlp/gate" || gate.rows != 12288 || gate.cols != 4096) {
            return fail("gate patch shape");
        }
        if (up.tensor_name != "dit/layers/0/mlp/up" || up.rows != 12288) { return fail("up patch shape"); }
        if (down.tensor_name != "dit/layers/0/mlp/down" || down.rows != 4096 || down.cols != 12288) {
            return fail("down patch shape");
        }
        for (int layer = 0; layer < 32; ++layer) {
            for (LoraSlot slot : {LoraSlot::Query, LoraSlot::Key, LoraSlot::Value, LoraSlot::Output, LoraSlot::Gate,
                                  LoraSlot::Up, LoraSlot::Down}) {
                (void)find_patch(patches, layer, slot);
            }
        }
        print_element("to_q", query, 0, 0, 1.F, 0);
        print_element("to_q", query, 0, 1, 1.F, 0);
        print_element("to_q", query, 3, 5, 0.5F, ninfer_bf16_from_f32(1.F));
        print_element("gate", gate, 0, 0, 1.F, 0);
        print_element("up", up, 0, 0, 1.F, 0);
        print_element("down", down, 1, 2, 1.F, 0);

        const fs::path tmp = fs::temp_directory_path() / "ninfer-lora-host-check.safetensors";
        const std::string header =
            R"({"__metadata__":{"note":"quote \" inside"},"not.a.lora":{"dtype":"BF16","shape":[1,1],"data_offsets":[0,2]},)"
            R"("diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight":{"dtype":"BF16","shape":[1,4096],"data_offsets":[0,8192]},)"
            R"("diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight":{"dtype":"BF16","shape":[4096,1],"data_offsets":[8192,16384]}})";
        std::vector<unsigned char> file(8 + header.size() + 16384);
        const std::uint64_t header_len = header.size();
        for (int i = 0; i < 8; ++i) { file[i] = static_cast<unsigned char>((header_len >> (8 * i)) & 0xFF); }
        for (std::size_t i = 0; i < header.size(); ++i) { file[8 + i] = static_cast<unsigned char>(header[i]); }
        const std::uint16_t one = ninfer_bf16_from_f32(1.F);
        for (int i = 0; i < 8192; ++i) {
            const std::size_t at = 8 + header.size() + static_cast<std::size_t>(i) * 2;
            file[at]     = static_cast<unsigned char>(one & 0xFF);
            file[at + 1] = static_cast<unsigned char>((one >> 8) & 0xFF);
        }
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
        }
        const auto tmp_hash = fnv(tmp);
        const std::vector<LoraPatch> one_pair = read_lora_patches(tmp);
        if (fnv(tmp) != tmp_hash) { return fail("synthetic LoRA file changed"); }
        if (one_pair.size() != 1 || one_pair[0].rank != 1 || one_pair[0].a[0] != one || one_pair[0].b[0] != one) {
            return fail("synthetic LoRA parse");
        }
        const std::uint16_t added = ninfer_lora_element(0, one_pair[0].b.data(), one_pair[0].a.data(), one_pair[0].cols,
                                                       one_pair[0].rank, 0, 1.F);
        if (added != one) { return fail("synthetic scale * (B @ A)"); }
        fs::remove(tmp);

        bool threw = false;
        try {
            const std::vector<LoraPatch> missing = read_lora_patches(tmp);
            if (!missing.empty()) { return fail("missing file should fail"); }
        } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) { return fail("missing file should fail"); }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& e) {
        return fail(e.what());
    }
}
