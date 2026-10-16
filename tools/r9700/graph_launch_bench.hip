// Measures CPU submission separately from synchronized GPU completion. Alternating immutable
// graphs retain distinct 512-byte arguments; an independent integer sum checks every replay.
#include "core/device.h"
#include "qualification_device_guard.h"

#include <hip/hip_runtime.h>

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <time.h>
#include <vector>

namespace {

struct Arguments {
    std::uint64_t words[64];
};

__global__ void advance(std::uint64_t* output, Arguments arguments) {
    if (threadIdx.x == 0) *output += arguments.words[0] + arguments.words[63];
}

double cpu_time(clockid_t clock) {
    timespec value{};
    if (clock_gettime(clock, &value) != 0) throw std::runtime_error("clock_gettime failed");
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_nsec) * 1e-9;
}

std::uint32_t positive_number(std::string_view text) {
    std::uint32_t value = 0;
    const auto parsed   = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0)
        throw std::runtime_error("expected positive node/graph count");
    return value;
}

std::size_t hip_free() {
    std::size_t available = 0, total = 0;
    HIP_CHECK(hipMemGetInfo(&available, &total));
    return available;
}

std::int64_t difference(std::size_t before, std::size_t after) {
    return static_cast<std::int64_t>(before) - static_cast<std::int64_t>(after);
}

} // namespace

int main(int argc, char** argv) try {
    std::uint32_t nodes = 1000, graph_count = 2;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("use --nodes N --graphs N");
        const std::string_view option(argv[i]);
        const auto value = positive_number(argv[i + 1]);
        if (option == "--nodes")
            nodes = value;
        else if (option == "--graphs")
            graph_count = value;
        else
            throw std::runtime_error("use --nodes N --graphs N");
    }
    auto& guard = ninfer::r9700_qualification::device_guard();
    ninfer::DeviceContext device(guard.select());
    std::uint64_t* output = nullptr;
    HIP_CHECK(hipMalloc(&output, sizeof(*output)));
    HIP_CHECK(hipMemsetAsync(output, 0, sizeof(*output), device.stream));
    advance<<<1, 32, 0, device.stream>>>(output, Arguments{});
    device.synchronize_all();
    std::uint64_t initialized = 0;
    HIP_CHECK(hipMemcpy(&initialized, output, sizeof(initialized), hipMemcpyDeviceToHost));
    if (initialized != 0) throw std::runtime_error("integer initialization oracle mismatch");

    std::vector<hipGraph_t> graphs(graph_count);
    std::vector<hipGraphExec_t> executables(graph_count);
    std::vector<std::uint64_t> increments(graph_count);
    const auto before_graphs           = hip_free();
    const auto available_before_graphs = device.available_vram();
    for (std::uint32_t g = 0; g < graph_count; ++g) {
        HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeThreadLocal));
        for (std::uint32_t n = 0; n < nodes; ++n) {
            Arguments arguments{};
            arguments.words[0]  = n % 7U + g + 1U;
            arguments.words[63] = n % 11U + 1U;
            increments[g] += arguments.words[0] + arguments.words[63];
            advance<<<1, 32, 0, device.stream>>>(output, arguments);
        }
        HIP_CHECK(hipStreamEndCapture(device.stream, &graphs[g]));
        HIP_CHECK(hipGraphInstantiate(&executables[g], graphs[g], nullptr, nullptr, 0));
        HIP_CHECK(hipGraphUpload(executables[g], device.stream));
    }
    device.synchronize_all();
    std::uint64_t captured = 0;
    HIP_CHECK(hipMemcpy(&captured, output, sizeof(captured), hipMemcpyDeviceToHost));
    if (captured != 0) throw std::runtime_error("graph capture changed integer output");
    const auto after_graphs           = hip_free();
    const auto available_after_graphs = device.available_vram();
    std::uint64_t reference           = 0;
    for (std::uint32_t i = 0; i < 10U + graph_count; ++i) {
        HIP_CHECK(hipGraphLaunch(executables[i % graph_count], device.stream));
        reference += increments[i % graph_count];
        device.synchronize();
    }
    device.synchronize_all();
    std::uint64_t warmed = 0;
    HIP_CHECK(hipMemcpy(&warmed, output, sizeof(warmed), hipMemcpyDeviceToHost));
    if (warmed != reference) {
        std::cerr << "warmup observed=" << warmed << " expected=" << reference << '\n';
        throw std::runtime_error("integer warmup oracle mismatch");
    }
    guard.check("after-exact-warmup");

    std::cout << "{\"nodes\":" << nodes << ",\"graphs\":" << graph_count
              << ",\"payload_bytes\":" << sizeof(Arguments)
              << ",\"argument_bytes\":" << sizeof(Arguments) + sizeof(output)
              << ",\"hip_graph_bytes\":" << difference(before_graphs, after_graphs)
              << ",\"available_graph_bytes\":"
              << difference(available_before_graphs, available_after_graphs) << ",\"reps\":[";
    constexpr std::uint32_t launches = 100;
    for (int repetition = 0; repetition < 7; ++repetition) {
        double thread_cpu = 0, enqueue = 0;
        const double process_begin = cpu_time(CLOCK_PROCESS_CPUTIME_ID);
        const auto begin           = std::chrono::steady_clock::now();
        for (std::uint32_t i = 0; i < launches; ++i) {
            const double cpu_begin  = cpu_time(CLOCK_THREAD_CPUTIME_ID);
            const auto launch_begin = std::chrono::steady_clock::now();
            HIP_CHECK(hipGraphLaunch(executables[i % graph_count], device.stream));
            const auto launch_end = std::chrono::steady_clock::now();
            thread_cpu += cpu_time(CLOCK_THREAD_CPUTIME_ID) - cpu_begin;
            enqueue += std::chrono::duration<double>(launch_end - launch_begin).count();
            reference += increments[i % graph_count];
            device.synchronize();
        }
        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        const double process_cpu = cpu_time(CLOCK_PROCESS_CPUTIME_ID) - process_begin;
        std::uint64_t observed   = 0;
        HIP_CHECK(hipMemcpy(&observed, output, sizeof(observed), hipMemcpyDeviceToHost));
        if (observed != reference) throw std::runtime_error("integer replay oracle mismatch");
        if (repetition != 0) std::cout << ',';
        constexpr double to_microseconds_per_launch = 1e6 / launches;
        std::cout << "{\"thread_cpu_us\":" << thread_cpu * to_microseconds_per_launch
                  << ",\"enqueue_us\":" << enqueue * to_microseconds_per_launch
                  << ",\"wall_us\":" << wall * to_microseconds_per_launch
                  << ",\"process_cpu_us\":" << process_cpu * to_microseconds_per_launch
                  << ",\"exact\":true}" << std::flush;
    }
    std::cout << "],\"hip_late_bytes\":" << difference(after_graphs, hip_free()) << "}\n";
    guard.check("after-timing");
    for (auto executable : executables) HIP_CHECK(hipGraphExecDestroy(executable));
    for (auto graph : graphs) HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipFree(output));
    return 0;
} catch (const std::exception& error) {

    std::cerr << error.what() << '\n';
    return 1;
}
