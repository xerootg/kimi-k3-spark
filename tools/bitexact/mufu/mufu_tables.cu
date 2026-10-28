// Generate the MUFU tables on the GB10 and verify mufu_emu.h against the hardware over all 2^32 inputs.
// nvcc -O3 -use_fast_math -arch=sm_121a mufu_tables.cu -o mufu_tables   (the same flags as ggml-cuda)
// usage: mufu_tables gen DIR | mufu_tables verify DIR
// The hardware ops are emitted with the PTX that ggml-cuda's code compiles to (tanh.approx.f32, ex2.approx.ftz.f32,
// rcp.approx.ftz.f32); `cuobjdump -sass` of this binary shows exactly one MUFU.TANH / MUFU.EX2 / MUFU.RCP each.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#define MUFU_EMU_FN static __host__ __device__ inline
#include "mufu_emu.h"

__device__ __forceinline__ float hw_tanh(float x) { float y; asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float hw_ex2(float x)  { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float hw_rcp(float x)  { float y; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

__global__ void gen_k(int op, uint32_t base, uint32_t * out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;   // 0 .. 2^23-1
    uint32_t u = base | i;
    float x = __uint_as_float(u);
    out[i] = __float_as_uint(op == 0 ? hw_rcp(x) : op == 1 ? hw_tanh(x) : hw_ex2(x));
}

struct Bad { unsigned long long n; uint32_t first[8], got[8], want[8]; };

__global__ void verify_k(int op, uint32_t base, mufu_tables T, Bad * bad) {
    uint32_t u = base + blockIdx.x * blockDim.x + threadIdx.x;
    float x = __uint_as_float(u);
    uint32_t hw, em;
    if (op == 0) { hw = __float_as_uint(hw_rcp(x)); em = mufu_rcp_bits(&T, u); }
    else if (op == 1) { hw = __float_as_uint(hw_tanh(x)); em = mufu_tanh_bits(&T, u); }
    else { hw = __float_as_uint(hw_ex2(x)); em = mufu_ex2_bits(&T, u); }
    if (hw != em) {
        unsigned long long k = atomicAdd(&bad->n, 1ull);
        if (k < 8) { bad->first[k] = u; bad->got[k] = em; bad->want[k] = hw; }
    }
}

static void gen_file(const std::string & path, int op, uint32_t sign, uint32_t e0, uint32_t e1, uint32_t * dbuf, std::vector<uint32_t> & h) {
    FILE * f = fopen(path.c_str(), "wb");
    for (uint32_t E = e0; E <= e1; E++) {
        gen_k<<<(1 << 23) / 256, 256>>>(op, sign | (E << 23), dbuf);
        cudaMemcpy(h.data(), dbuf, 4u << 23, cudaMemcpyDeviceToHost);
        fwrite(h.data(), 4, 1u << 23, f);
    }
    fclose(f);
    printf("wrote %s (%u binades)\n", path.c_str(), e1 - e0 + 1);
}

static uint32_t * load(const std::string & path, size_t n) {
    std::vector<uint32_t> h(n);
    FILE * f = fopen(path.c_str(), "rb");
    if (!f || fread(h.data(), 4, n, f) != n) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    fclose(f);
    uint32_t * d; cudaMalloc(&d, n * 4); cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice);
    return d;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: mufu_tables gen|verify DIR\n"); return 1; }
    std::string mode = argv[1], dir = argv[2];
    if (mode == "gen") {
        uint32_t * d; cudaMalloc(&d, 4u << 23);
        std::vector<uint32_t> h(1u << 23);
        gen_file(dir + "/rcp.bin",  0, 0, 127, 127, d, h);
        gen_file(dir + "/tanh.bin", 1, 0, MUFU_TANH_E0, MUFU_TANH_E1, d, h);
        gen_file(dir + "/ex2p.bin", 2, 0, MUFU_EX2P_E0, MUFU_EX2P_E1, d, h);
        gen_file(dir + "/ex2n.bin", 2, 0x80000000u, MUFU_EX2N_E0, MUFU_EX2N_E1, d, h);
        cudaFree(d);
        printf("err %s\n", cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    const char * names[3] = {"rcp", "tanh", "ex2"};
    Bad * bad; cudaMalloc(&bad, sizeof(Bad));
    int fails = 0;
    for (int op = 0; op < 3; op++) {
        mufu_tables T = {nullptr, nullptr, nullptr, nullptr};
        if (op == 0) T.rcp = load(dir + "/rcp.bin", 1u << 23);
        if (op == 1) T.tanh_ = load(dir + "/tanh.bin", (size_t) (MUFU_TANH_E1 - MUFU_TANH_E0 + 1) << 23);
        if (op == 2) {
            T.ex2p = load(dir + "/ex2p.bin", (size_t) (MUFU_EX2P_E1 - MUFU_EX2P_E0 + 1) << 23);
            T.ex2n = load(dir + "/ex2n.bin", (size_t) (MUFU_EX2N_E1 - MUFU_EX2N_E0 + 1) << 23);
        }
        cudaMemset(bad, 0, sizeof(Bad));
        for (uint64_t b = 0; b < (1ull << 32); b += (1ull << 26)) verify_k<<<(1 << 26) / 256, 256>>>(op, (uint32_t) b, T, bad);
        Bad h; cudaMemcpy(&h, bad, sizeof h, cudaMemcpyDeviceToHost);
        printf("%s: %llu mismatches over 2^32 inputs\n", names[op], h.n);
        for (unsigned i = 0; i < h.n && i < 8; i++) printf("   x=%08x emu=%08x hw=%08x\n", h.first[i], h.got[i], h.want[i]);
        fails += h.n != 0;
        cudaFree((void *) T.rcp); cudaFree((void *) T.tanh_); cudaFree((void *) T.ex2p); cudaFree((void *) T.ex2n);
    }
    printf("err %s\n", cudaGetErrorString(cudaGetLastError()));
    return fails;
}
