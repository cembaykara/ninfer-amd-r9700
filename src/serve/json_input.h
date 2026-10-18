#pragma once

#include <nlohmann/json.hpp>

#include <string_view>

namespace ninfer::serve {

// Parse once, validating schema numbers before ordinary DOM forwarding loses their spelling.
nlohmann::json parse_request_body(std::string_view source);

} // namespace ninfer::serve
