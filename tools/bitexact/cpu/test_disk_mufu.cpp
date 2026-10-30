// Linked ONLY into test executables via --wrap; no production loader bypass.
#include "k3cpu.h"
#include <cstdio>
#include <vector>

// A real packed table with deterministic constant mantissa reciprocal 0.75.
// EX2 and TANH use one-entry packed tables and the existing boundary rules.
// This exercises packed lookup, activation, quantization and real row kernels,
// but does NOT claim to reproduce NVIDIA's unavailable exhaustive MUFU tables.
bool fixture_load(const char *, k3_mufu_tabs * out)
    asm("__wrap__Z15k3cpu_load_mufuPKcP12k3_mufu_tabs");
bool fixture_load(const char *, k3_mufu_tabs * out) {
    static std::vector<uint32_t> hdr((1u << 23) / 64 * 2);
    static std::vector<int8_t> res(1u << 23, 0);
    for (size_t i = 0; i < hdr.size(); i += 2) hdr[i] = 0x3f400000u;
    static const uint32_t one_hdr[] = {0x3f000000u, 0};
    static const int8_t one_res[] = {0};
    const k3_ptab rcp{hdr.data(), res.data(), nullptr, 0, 1u << 23};
    const k3_ptab tiny{one_hdr, one_res, nullptr, 0, 1};
    *out = k3_mufu_tabs{rcp, tiny, tiny, tiny, 1, 1, 1};
    std::fprintf(stderr, "TEST ONLY: deterministic synthetic packed MUFU fixture\n");
    return true;
}
