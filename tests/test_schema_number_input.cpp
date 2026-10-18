#include "serve/json_input.h"
#include "serve/request.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        for (const auto& [prefix, suffix] :
             {std::pair{R"({"response_format":{"json_schema":{"schema":)", "}}}"},
              std::pair{R"({"text":{"format":{"schema":)", "}}}"},
              std::pair{R"({"output_config":{"format":{"schema":)", "}}}"},
              std::pair{R"({"tools":[{"function":{"parameters":)", "}}]}"},
              std::pair{R"({"tools":[{"parameters":)", "}]}"},
              std::pair{R"({"tools":[{"input_schema":)", "}]}"}}) {
            const auto bad = std::string(prefix) + R"({"const":9007199254740993.0})" + suffix;
            bool rejected  = false;
            try {
                (void)ninfer::serve::parse_request_body(bad);
            } catch (const ninfer::serve::ApiException& error) {
                rejected = error.error().status == 400 && error.error().param.ends_with("/const") &&
                           error.error().message.find(error.error().param) != std::string::npos &&
                           error.error().code == (std::string_view(prefix).starts_with("{\"tools\"")
                                                      ? "invalid_tool_schema"
                                                      : "invalid_output_constraint");
            }
            if (!rejected) throw std::runtime_error("request schema silently rounded: " + bad);
            const auto exact = std::string(prefix) + R"({"const":9007199254740993})" + suffix;
            (void)ninfer::serve::parse_request_body(exact);
        }
        (void)ninfer::serve::parse_request_body(R"({"temperature":1.00000000000000001})");
        std::cout << "PASS: HTTP schema numeric-source boundaries\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
