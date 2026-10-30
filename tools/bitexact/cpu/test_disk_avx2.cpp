// Validate the test-only integer instruction substitution against a scalar dot.
#include "test_disk_avx2.h"
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    uint32_t rng = 0x91827364;
    for (unsigned trial = 0; trial < 10000; ++trial) {
        alignas(32) uint8_t a[32];
        alignas(32) int8_t b[32];
        alignas(32) uint32_t acc[8], got[8];
        auto next = [&] { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
        for (int i = 0; i < 32; ++i) {
            a[i] = trial < 256 ? trial : next();
            b[i] = trial < 256 ? int8_t(i % 2 ? -128 : 127) : int8_t(next());
        }
        for (auto & x : acc) x = next();
        _mm256_store_si256(reinterpret_cast<__m256i *>(got), disk_test_dpbusd(
            _mm256_load_si256(reinterpret_cast<const __m256i *>(acc)),
            _mm256_load_si256(reinterpret_cast<const __m256i *>(a)),
            _mm256_load_si256(reinterpret_cast<const __m256i *>(b))));
        for (int lane = 0; lane < 8; ++lane) {
            uint32_t expected = acc[lane];
            for (int j = 0; j < 4; ++j) expected += int(a[lane * 4 + j]) * int(b[lane * 4 + j]);
            if (got[lane] != expected) {
                std::fprintf(stderr, "AVX2 integer-dot mismatch at trial %u lane %d\n", trial, lane);
                return 1;
            }
        }
    }
    std::puts("PASS exact AVX2 integer-dot substitution (80000 lanes, signed extremes/overflow)");
}
