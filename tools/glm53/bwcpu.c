// bwcpu: CPU streaming-read bandwidth with T threads (AVX2 loads over N MB), best of 5
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int c, char ** v) {
    size_t n = (size_t) atoi(v[1]) << 20; int T = atoi(v[2]);
    char * a = aligned_alloc(64, n); memset(a, 1, n); double best = 1e9; long long sink = 0;
    for (int r = 0; r < 5; r++) {
        double t0 = omp_get_wtime();
        #pragma omp parallel num_threads(T) reduction(+:sink)
        { int t = omp_get_thread_num(); size_t ch = n / T / 128 * 128; __m256i s = _mm256_setzero_si256();
          for (size_t i = t * ch; i < (t + 1) * ch; i += 128) { s = _mm256_xor_si256(s, _mm256_load_si256((void *) (a + i)));
            s = _mm256_xor_si256(s, _mm256_load_si256((void *) (a + i + 32))); s = _mm256_xor_si256(s, _mm256_load_si256((void *) (a + i + 64)));
            s = _mm256_xor_si256(s, _mm256_load_si256((void *) (a + i + 96))); }
          sink += _mm256_extract_epi64(s, 0); }
        double t = omp_get_wtime() - t0; if (t < best) best = t;
    }
    printf("threads %d: %.1f GB/s (%lld)\n", T, n / best / 1e9, sink & 1);
}
