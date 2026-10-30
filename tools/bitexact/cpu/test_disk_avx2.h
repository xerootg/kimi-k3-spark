// Test-only, exact unsigned-byte x signed-byte dot on CPUs without AVX-VNNI.
// Splitting even/odd bytes avoids the saturation of a single maddubs.
#pragma once
#include <immintrin.h>
static inline __m256i disk_test_dpbusd(__m256i acc, __m256i a, __m256i b) {
    const __m256i mask = _mm256_set1_epi16(0x00ff);
    const __m256i even = _mm256_maddubs_epi16(_mm256_and_si256(a, mask), b);
    const __m256i odd = _mm256_maddubs_epi16(_mm256_andnot_si256(mask, a), b);
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_add_epi32(acc, _mm256_add_epi32(
        _mm256_madd_epi16(even, ones), _mm256_madd_epi16(odd, ones)));
}
#define _mm256_dpbusd_avx_epi32 disk_test_dpbusd
