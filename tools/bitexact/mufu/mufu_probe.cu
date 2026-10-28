// Per-binade survey of the GB10's MUFU.TANH / MUFU.EX2 / MUFU.RCP against correctly rounded references.
// nvcc -O3 -use_fast_math -arch=sm_121a mufu_probe.cu -o mufu_probe
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

__device__ __forceinline__ float mufu_tanh(float x) { float y; asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float mufu_ex2(float x)  { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float mufu_rcp(float x)  { float y; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// stats per (op, binade = top 9 bits of x): n_exact, n_diff, min ulp diff, max ulp diff (vs RN of double ref)
struct St { unsigned long long nexact, ndiff; int mind, maxd; };

__device__ int ulpdiff(float a, float b) { // a - b in ulps (ordered ints); both finite same sign assumed
    int ia = __float_as_int(a), ib = __float_as_int(b);
    if (ia < 0) ia = 0x80000000 - ia;
    if (ib < 0) ib = 0x80000000 - ib;
    long long d = (long long) ia - ib;
    if (d > 1000000) d = 1000000; if (d < -1000000) d = -1000000;
    return (int) d;
}

__global__ void k(uint32_t base, St * st) {
    uint32_t u = base + blockIdx.x * blockDim.x + threadIdx.x;
    float x = __uint_as_float(u);
    int bin = u >> 23;
    float r[3];
    r[0] = mufu_tanh(x); r[1] = mufu_ex2(x); r[2] = mufu_rcp(x);
    double xd = (double) x;
    float ref[3];
    ref[0] = (float) tanh(xd); ref[1] = (float) exp2(xd); ref[2] = (float) (1.0 / xd);
    for (int o = 0; o < 3; o++) {
        St * s = &st[o * 512 + bin];
        if (__float_as_uint(r[o]) == __float_as_uint(ref[o])) { atomicAdd(&s->nexact, 1ull); }
        else {
            atomicAdd(&s->ndiff, 1ull);
            int d = (isfinite(r[o]) && isfinite(ref[o])) ? ulpdiff(r[o], ref[o]) : 999999;
            atomicMin(&s->mind, d); atomicMax(&s->maxd, d);
        }
    }
}

int main() {
    St * st; cudaMalloc(&st, sizeof(St) * 3 * 512);
    St h0[3 * 512]; for (auto & s : h0) { s.nexact = s.ndiff = 0; s.mind = 1 << 30; s.maxd = -(1 << 30); }
    cudaMemcpy(st, h0, sizeof h0, cudaMemcpyHostToDevice);
    for (uint64_t b = 0; b < (1ull << 32); b += (1ull << 26)) k<<<(1 << 26) / 256, 256>>>((uint32_t) b, st);
    cudaMemcpy(h0, st, sizeof h0, cudaMemcpyDeviceToHost);
    const char * nm[3] = {"tanh", "ex2", "rcp"};
    for (int o = 0; o < 3; o++) for (int b = 0; b < 512; b++) {
        St & s = h0[o * 512 + b];
        if (s.ndiff) printf("%s bin %3d (sign %d exp %3d): exact %llu diff %llu ulp [%d, %d]\n", nm[o], b, b >> 8, (b & 255) - 127, s.nexact, s.ndiff, s.mind, s.maxd);
    }
    printf("err %s\n", cudaGetErrorString(cudaGetLastError()));
}
