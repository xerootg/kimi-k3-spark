// Structure probes for MUFU.EX2 (fixed-point table hypotheses), MUFU.RCP (mantissa table) and MUFU.TANH
// (symmetry, trivial ranges, monotonicity / step sizes).
#include <cstdio>
#include <cstdint>
#include <cmath>

__device__ __forceinline__ float mufu_tanh(float x) { float y; asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float mufu_ex2(float x)  { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float mufu_rcp(float x)  { float y; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

__global__ void build_ex2(uint32_t * T, int k) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < (1u << k)) T[f] = __float_as_uint(mufu_ex2(ldexpf((float) f, -k)));
}
// counters: [k*3+mode] mismatches over finite x in (-126, 128) whose prediction is a normal float
__global__ void test_ex2(uint32_t base, const uint32_t * T22, const uint32_t * T23, const uint32_t * T24, unsigned long long * cnt, unsigned long long * ntest) {
    uint32_t u = base + blockIdx.x * blockDim.x + threadIdx.x;
    float x = __uint_as_float(u);
    if (!(fabsf(x) < 125.0f)) return;
    if (fabsf(x) < 1e-30f) return;
    uint32_t r = __float_as_uint(mufu_ex2(x));
    atomicAdd(ntest, 1ull);
    const uint32_t * Ts[3] = {T22, T23, T24};
    for (int kk = 0; kk < 3; kk++) {
        int k = 22 + kk;
        double xs = ldexp((double) x, k); // exact
        long long X[3] = { (long long) rint(xs), (long long) trunc(xs), (long long) floor(xs) };
        for (int m = 0; m < 3; m++) {
            long long i = X[m] >> k; uint32_t f = (uint32_t) (X[m] & ((1ll << k) - 1));
            uint32_t t = Ts[kk][f];
            int e = (int) ((t >> 23) & 255) + (int) i;
            uint32_t p = e >= 1 && e <= 254 ? (t & 0x807fffff) | ((uint32_t) e << 23) : 0xffffffff;
            if (p != r) atomicAdd(&cnt[kk * 3 + m], 1ull);
        }
    }
}

__global__ void build_rcp(uint32_t * T) {
    uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < (1u << 23)) T[m] = __float_as_uint(mufu_rcp(__uint_as_float(0x3f800000u | m)));
}
// rcp model: normal x with biased exponent E: result = T[m] with exponent shifted by (127 - E), when that stays normal
__global__ void test_rcp(uint32_t base, const uint32_t * T, unsigned long long * cnt) {
    uint32_t u = base + blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t E = (u >> 23) & 255, m = u & 0x7fffff;
    if (E == 0 || E == 255) return;
    uint32_t r = __float_as_uint(mufu_rcp(__uint_as_float(u)));
    uint32_t t = T[m];
    int e = (int) ((t >> 23) & 255) + 127 - (int) E;
    if (e >= 1 && e <= 254) {
        uint32_t p = (t & 0x7fffff) | ((uint32_t) e << 23) | (u & 0x80000000u);
        atomicAdd(&cnt[p == r ? 0 : 1], 1ull);
    } else {
        atomicAdd(&cnt[2], 1ull);  // out of normal range: record what happens
        if (r != 0 && (r & 0x7fffffff) != 0) atomicAdd(&cnt[3], 1ull); // nonzero (denormal) output
    }
}

// tanh: odd symmetry; trivial ranges; per positive binade: monotone? max step in output bits between neighbours
__global__ void test_tanh(uint32_t base, unsigned long long * cnt, int * maxstep, int * minstep) {
    uint32_t u = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= 0x7f800000u) return; // positive finite only
    float x = __uint_as_float(u);
    uint32_t r = __float_as_uint(mufu_tanh(x));
    uint32_t rn = __float_as_uint(mufu_tanh(-x));
    if (rn != (r ^ 0x80000000u)) atomicAdd(&cnt[0], 1ull);
    int bin = u >> 23;
    if (r == u) atomicAdd(&cnt[1], 1ull);            // tanh(x) == x
    if (r == 0x3f800000u) atomicAdd(&cnt[2], 1ull);  // == 1
    if (u + 1 < 0x7f800000u && ((u + 1) >> 23) == (uint32_t) bin) {
        int d = (int) __float_as_uint(mufu_tanh(__uint_as_float(u + 1))) - (int) r;
        atomicMax(&maxstep[bin], d); atomicMin(&minstep[bin], d);
    }
}

int main() {
    uint32_t *T22, *T23, *T24, *TR;
    cudaMalloc(&T22, 4u << 22); cudaMalloc(&T23, 4u << 23); cudaMalloc(&T24, 4u << 24); cudaMalloc(&TR, 4u << 23);
    build_ex2<<<(1 << 22) / 256, 256>>>(T22, 22);
    build_ex2<<<(1 << 23) / 256, 256>>>(T23, 23);
    build_ex2<<<(1 << 24) / 256, 256>>>(T24, 24);
    build_rcp<<<(1 << 23) / 256, 256>>>(TR);
    unsigned long long * c; cudaMalloc(&c, 64 * 8); cudaMemset(c, 0, 64 * 8);
    int * mx; int * mn; cudaMalloc(&mx, 256 * 4); cudaMalloc(&mn, 256 * 4);
    int h[256]; for (int i = 0; i < 256; i++) h[i] = -(1 << 30); cudaMemcpy(mx, h, sizeof h, cudaMemcpyHostToDevice);
    for (int i = 0; i < 256; i++) h[i] = 1 << 30; cudaMemcpy(mn, h, sizeof h, cudaMemcpyHostToDevice);
    for (uint64_t b = 0; b < (1ull << 32); b += (1ull << 26)) {
        test_ex2<<<(1 << 26) / 256, 256>>>((uint32_t) b, T22, T23, T24, c, c + 10);
        test_rcp<<<(1 << 26) / 256, 256>>>((uint32_t) b, TR, c + 16);
        test_tanh<<<(1 << 26) / 256, 256>>>((uint32_t) b, c + 24, mx, mn);
    }
    unsigned long long hc[64]; cudaMemcpy(hc, c, sizeof hc, cudaMemcpyDeviceToHost);
    printf("ex2 tested %llu\n", hc[10]);
    const char * mode[3] = {"RN", "RZ", "RD"};
    for (int k = 0; k < 3; k++) for (int m = 0; m < 3; m++) printf("ex2 k=%d %s mismatches %llu\n", 22 + k, mode[m], hc[k * 3 + m]);
    printf("rcp model ok %llu bad %llu outofrange %llu (nonzero %llu)\n", hc[16], hc[17], hc[18], hc[19]);
    printf("tanh asym %llu, ==x %llu, ==1 %llu\n", hc[24], hc[25], hc[26]);
    int hx[256], hn[256]; cudaMemcpy(hx, mx, sizeof hx, cudaMemcpyDeviceToHost); cudaMemcpy(hn, mn, sizeof hn, cudaMemcpyDeviceToHost);
    for (int b = 100; b < 140; b++) printf("tanh bin exp %4d: step [%d, %d]\n", b - 127, hn[b], hx[b]);
    printf("err %s\n", cudaGetErrorString(cudaGetLastError()));
}
