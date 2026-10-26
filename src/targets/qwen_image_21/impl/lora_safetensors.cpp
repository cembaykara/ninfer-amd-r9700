#include "lora_safetensors.h"

#include "config.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen_image_21::detail {
namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument("image_denoise: lora " + message);
}

struct HeaderTensor {
    int rows = 0;
    int cols = 0;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

class JsonCursor {
public:
    explicit JsonCursor(std::string text) : text_(std::move(text)) {}

    char peek() {
        skip_ws();
        if (i_ >= text_.size()) { fail("safetensors header ended early"); }
        return text_[i_];
    }

    char get() {
        const char c = peek();
        ++i_;
        return c;
    }

    void expect(char want) {
        const char got = get();
        if (got != want) { fail("safetensors header is not valid JSON"); }
    }

    bool consume(char want) {
        if (peek() != want) { return false; }
        ++i_;
        return true;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (i_ < text_.size()) {
            const char c = text_[i_++];
            if (c == '"') { return out; }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (i_ >= text_.size()) { fail("safetensors header has a truncated string"); }
            const char escaped = text_[i_++];
            switch (escaped) {
            case '"':
            case '\\':
            case '/':
                out.push_back(escaped);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u':
                append_unicode(out);
                break;
            default:
                fail("safetensors header has a bad string escape");
            }
        }
        fail("safetensors header has an unterminated string");
    }

    std::uint64_t parse_u64() {
        skip_ws();
        if (i_ >= text_.size() || text_[i_] < '0' || text_[i_] > '9') {
            fail("safetensors header expected a number");
        }
        std::uint64_t value = 0;
        while (i_ < text_.size() && text_[i_] >= '0' && text_[i_] <= '9') {
            const std::uint64_t digit = static_cast<std::uint64_t>(text_[i_] - '0');
            if (value > (static_cast<std::uint64_t>(-1) - digit) / 10) {
                fail("safetensors header number overflowed");
            }
            value = value * 10 + digit;
            ++i_;
        }
        return value;
    }

    void skip_value() {
        const char c = peek();
        if (c == '"') {
            (void)parse_string();
            return;
        }
        if (c == '{') {
            skip_object();
            return;
        }
        if (c == '[') {
            skip_array();
            return;
        }
        if (c == 't' || c == 'f' || c == 'n') {
            skip_literal();
            return;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            skip_number();
            return;
        }
        fail("safetensors header is not valid JSON");
    }

    void finish() {
        skip_ws();
        if (i_ != text_.size()) { fail("safetensors header has trailing data"); }
    }

private:
    void skip_ws() {
        while (i_ < text_.size()) {
            const char c = text_[i_];
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') { return; }
            ++i_;
        }
    }

    void append_unicode(std::string& out) {
        if (i_ + 4 > text_.size()) { fail("safetensors header has a short unicode escape"); }
        unsigned code = 0;
        for (int n = 0; n < 4; ++n) {
            const char h = text_[i_++];
            code <<= 4;
            if (h >= '0' && h <= '9') {
                code |= static_cast<unsigned>(h - '0');
            } else if (h >= 'a' && h <= 'f') {
                code |= static_cast<unsigned>(h - 'a' + 10);
            } else if (h >= 'A' && h <= 'F') {
                code |= static_cast<unsigned>(h - 'A' + 10);
            } else {
                fail("safetensors header has a bad unicode escape");
            }
        }
        if (code < 0x80u) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800u) {
            out.push_back(static_cast<char>(0xC0u | (code >> 6)));
            out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
        } else {
            out.push_back(static_cast<char>(0xE0u | (code >> 12)));
            out.push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
        }
    }

    void skip_object() {
        expect('{');
        if (consume('}')) { return; }
        for (;;) {
            (void)parse_string();
            expect(':');
            skip_value();
            if (consume('}')) { return; }
            expect(',');
        }
    }

    void skip_array() {
        expect('[');
        if (consume(']')) { return; }
        for (;;) {
            skip_value();
            if (consume(']')) { return; }
            expect(',');
        }
    }

    void skip_literal() {
        skip_ws();
        const std::string_view rest(text_.data() + i_, text_.size() - i_);
        for (const char* word : {"true", "false", "null"}) {
            const std::size_t n = std::char_traits<char>::length(word);
            if (rest.starts_with(word)) {
                i_ += n;
                return;
            }
        }
        fail("safetensors header is not valid JSON");
    }

    void skip_number() {
        skip_ws();
        if (i_ < text_.size() && text_[i_] == '-') { ++i_; }
        if (i_ >= text_.size() || text_[i_] < '0' || text_[i_] > '9') {
            fail("safetensors header expected a number");
        }
        while (i_ < text_.size() && text_[i_] >= '0' && text_[i_] <= '9') { ++i_; }
        if (i_ < text_.size() && text_[i_] == '.') {
            ++i_;
            while (i_ < text_.size() && text_[i_] >= '0' && text_[i_] <= '9') { ++i_; }
        }
        if (i_ < text_.size() && (text_[i_] == 'e' || text_[i_] == 'E')) {
            ++i_;
            if (i_ < text_.size() && (text_[i_] == '+' || text_[i_] == '-')) { ++i_; }
            while (i_ < text_.size() && text_[i_] >= '0' && text_[i_] <= '9') { ++i_; }
        }
    }

    std::string text_;
    std::size_t i_ = 0;
};

HeaderTensor parse_tensor_object(JsonCursor& json) {
    json.expect('{');
    bool saw_dtype = false;
    bool saw_shape = false;
    bool saw_offsets = false;
    std::string dtype;
    int rows = 0;
    int cols = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    if (!json.consume('}')) {
        for (;;) {
            const std::string key = json.parse_string();
            json.expect(':');
            if (key == "dtype") {
                dtype = json.parse_string();
                saw_dtype = true;
            } else if (key == "shape") {
                json.expect('[');
                const std::uint64_t r = json.parse_u64();
                json.expect(',');
                const std::uint64_t c = json.parse_u64();
                json.expect(']');
                if (r == 0 || c == 0 || r > 1000000 || c > 1000000) {
                    fail("tensor shape is not a LoRA matrix");
                }
                rows = static_cast<int>(r);
                cols = static_cast<int>(c);
                saw_shape = true;
            } else if (key == "data_offsets") {
                json.expect('[');
                begin = json.parse_u64();
                json.expect(',');
                end = json.parse_u64();
                json.expect(']');
                saw_offsets = true;
            } else {
                json.skip_value();
            }
            if (json.consume('}')) { break; }
            json.expect(',');
        }
    }
    if (!saw_dtype || !saw_shape || !saw_offsets) { fail("tensor is missing dtype, shape, or data_offsets"); }
    if (dtype != "BF16") { fail("tensor dtype must be BF16"); }
    if (end < begin) { fail("tensor data_offsets are reversed"); }
    HeaderTensor tensor;
    tensor.rows  = rows;
    tensor.cols  = cols;
    tensor.begin = begin;
    tensor.end   = end;
    return tensor;
}

bool split_lora_name(const std::string& name, int& layer, std::string& mid, char& which) {
    constexpr std::string_view kPrefix = "diffusion_model.transformer_blocks.";
    if (!name.starts_with(kPrefix)) { return false; }
    std::size_t i = kPrefix.size();
    if (i >= name.size() || name[i] < '0' || name[i] > '9') { return false; }
    int idx = 0;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        if (idx > 100000) { return false; }
        idx = idx * 10 + (name[i] - '0');
        ++i;
    }
    if (i >= name.size() || name[i] != '.') { return false; }
    ++i;
    constexpr std::string_view kA = ".lora_A.weight";
    constexpr std::string_view kB = ".lora_B.weight";
    if (name.ends_with(kA) && name.size() >= i + kA.size()) {
        mid   = name.substr(i, name.size() - i - kA.size());
        which = 'A';
    } else if (name.ends_with(kB) && name.size() >= i + kB.size()) {
        mid   = name.substr(i, name.size() - i - kB.size());
        which = 'B';
    } else {
        return false;
    }
    if (mid.empty()) { return false; }
    layer = idx;
    return true;
}

struct AbPair {
    HeaderTensor a;
    HeaderTensor b;
    bool have_a = false;
    bool have_b = false;
};

struct Expect {
    LoraSlot slot;
    int rows;
    int cols;
    bool split;
};

Expect expect_for(const std::string& mid) {
    const int h = DitConfig::hidden;
    const int m = DitConfig::intermediate;
    if (mid == "attn.to_q") { return {LoraSlot::Query, h, h, false}; }
    if (mid == "attn.to_k") { return {LoraSlot::Key, h, h, false}; }
    if (mid == "attn.to_v") { return {LoraSlot::Value, h, h, false}; }
    if (mid == "attn.to_out.0") { return {LoraSlot::Output, h, h, false}; }
    if (mid == "img_mlp.gate_up") { return {LoraSlot::Gate, m, h, true}; }
    // Separate gate/up (the Qwen-Image-2.1-Turbo DiT stores mlp/gate and mlp/up
    // as distinct tensors; ComfyUI LoRAs name them gate_layer / proj).
    if (mid == "img_mlp.gate_layer") { return {LoraSlot::Gate, m, h, false}; }
    if (mid == "img_mlp.proj") { return {LoraSlot::Up, m, h, false}; }
    if (mid == "img_mlp.out") { return {LoraSlot::Down, h, m, false}; }
    fail("unexpected tensor " + mid);
}

std::vector<std::uint16_t> read_bf16(std::ifstream& in, std::uint64_t data_origin, const HeaderTensor& tensor) {
    const std::uint64_t bytes = tensor.end - tensor.begin;
    const auto elements = static_cast<std::uint64_t>(tensor.rows) * static_cast<std::uint64_t>(tensor.cols);
    if (bytes != elements * 2) { fail("tensor byte count does not match its shape"); }
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(bytes));
    in.seekg(static_cast<std::streamoff>(data_origin + tensor.begin));
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (!in) { fail("tensor payload could not be read"); }
    std::vector<std::uint16_t> out(static_cast<std::size_t>(elements));
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<std::uint16_t>(raw[2 * i] | (static_cast<std::uint16_t>(raw[2 * i + 1]) << 8));
    }
    return out;
}

std::string tensor_name(int layer, LoraSlot slot) {
    return "dit/layers/" + std::to_string(layer) + "/" + lora_slot_suffix(slot);
}

} // namespace

const char* lora_slot_suffix(LoraSlot slot) noexcept {
    switch (slot) {
    case LoraSlot::Query:
        return "attention/query";
    case LoraSlot::Key:
        return "attention/key";
    case LoraSlot::Value:
        return "attention/value";
    case LoraSlot::Output:
        return "attention/output";
    case LoraSlot::Gate:
        return "mlp/gate";
    case LoraSlot::Up:
        return "mlp/up";
    case LoraSlot::Down:
        return "mlp/down";
    }
    return "";
}

std::vector<LoraPatch> read_lora_patches(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { fail("file cannot be read: " + path.string()); }
    unsigned char len_bytes[8] = {};
    in.read(reinterpret_cast<char*>(len_bytes), 8);
    if (!in) { fail("file is too small to be safetensors"); }
    std::uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) { header_len |= static_cast<std::uint64_t>(len_bytes[i]) << (8 * i); }
    constexpr std::uint64_t kMaxHeader = 64ull << 20;
    if (header_len == 0 || header_len > kMaxHeader) { fail("safetensors header length is not usable"); }
    std::string header(static_cast<std::size_t>(header_len), '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!in) { fail("safetensors header could not be read"); }
    const std::uint64_t data_origin = 8 + header_len;

    JsonCursor json(std::move(header));
    std::map<std::pair<int, std::string>, AbPair> groups;
    json.expect('{');
    if (!json.consume('}')) {
        for (;;) {
            const std::string key = json.parse_string();
            json.expect(':');
            int layer = 0;
            std::string mid;
            char which = 0;
            if (key == "__metadata__" || !split_lora_name(key, layer, mid, which)) {
                json.skip_value();
            } else {
                HeaderTensor tensor = parse_tensor_object(json);
                AbPair& pair = groups[{layer, mid}];
                if (which == 'A') {
                    if (pair.have_a) { fail("duplicate A tensor " + key); }
                    pair.a = tensor;
                    pair.have_a = true;
                } else {
                    if (pair.have_b) { fail("duplicate B tensor " + key); }
                    pair.b = tensor;
                    pair.have_b = true;
                }
            }
            if (json.consume('}')) { break; }
            json.expect(',');
        }
    }
    json.finish();
    if (groups.empty()) { fail("file has no adapter tensors"); }

    in.seekg(0, std::ios::end);
    const std::streamoff end_pos = in.tellg();
    if (!in || end_pos < 0) { fail("file length could not be read"); }
    const auto file_end = static_cast<std::uint64_t>(end_pos);
    if (file_end < data_origin) { fail("safetensors file has no tensor payload"); }
    std::vector<LoraPatch> patches;
    for (const auto& [key, pair] : groups) {
        const int layer = key.first;
        const std::string& mid = key.second;
        if (!pair.have_a || !pair.have_b) { fail("missing A or B for block " + std::to_string(layer) + " " + mid); }
        if (layer < 0 || layer >= DitConfig::layers) { fail("block index is outside the DiT"); }
        const Expect expect = expect_for(mid);
        const int rank = pair.a.rows;
        if (rank <= 0 || pair.b.cols != rank || pair.a.cols != expect.cols) {
            fail("A/B shapes do not match " + mid);
        }
        const int b_rows = expect.split ? expect.rows * 2 : expect.rows;
        if (pair.b.rows != b_rows) { fail("B rows do not match " + mid); }
        if (pair.a.end > file_end - data_origin || pair.b.end > file_end - data_origin) {
            fail("tensor payload extends past the file");
        }
        std::vector<std::uint16_t> a = read_bf16(in, data_origin, pair.a);
        std::vector<std::uint16_t> b = read_bf16(in, data_origin, pair.b);

        const auto emit = [&](LoraSlot slot, int row0, int rows) {
            LoraPatch patch;
            patch.layer = layer;
            patch.slot  = slot;
            patch.rows  = rows;
            patch.cols  = expect.cols;
            patch.rank  = rank;
            patch.a     = a;
            patch.b.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(rank));
            for (int r = 0; r < rows; ++r) {
                const std::size_t src = (static_cast<std::size_t>(row0) + static_cast<std::size_t>(r)) * rank;
                std::memcpy(patch.b.data() + static_cast<std::size_t>(r) * rank, b.data() + src,
                            static_cast<std::size_t>(rank) * sizeof(std::uint16_t));
            }
            patch.tensor_name = tensor_name(layer, slot);
            patches.push_back(std::move(patch));
        };
        if (expect.split) {
            emit(LoraSlot::Gate, 0, expect.rows);
            emit(LoraSlot::Up, expect.rows, expect.rows);
        } else {
            emit(expect.slot, 0, expect.rows);
        }
    }
    return patches;
}

} // namespace ninfer::targets::qwen_image_21::detail
