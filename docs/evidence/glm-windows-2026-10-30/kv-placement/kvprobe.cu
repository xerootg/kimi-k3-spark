// kvprobe: can GPU kernels use KV-cache-shaped buffers in host memory, and what do Flash's decode access patterns
// cost there? (kimi-k3-spark patch 0051 companion, 2026-11-08). Three placements of one buffer:
//   dev  cudaMalloc (HBM on the GH200; the same LPDDR5X on the GB10)
//   reg  mmap + MADV_HUGEPAGE + mbind to the CPU nodes + populate + cudaHostRegister (what patch 0051 does with
//        4 KiB pages)
//   regnothp  the same without MADV_HUGEPAGE (patch 0051's default with 64 KiB pages, e.g. the GH200)
//   halloc cudaHostAlloc(Mapped|Portable)
// Checks: device pointer == host pointer, cudaMemset / cudaMemcpy (explicit kinds and Default) on the buffer, kernel
// read and write. Benchmarks, per decode token of GLM-5.3-Flash (11 attention layers):
//   gather4k  the flash-attention gather: 512 runs of 4 consecutive 1 KiB rows (pools of 4 tokens) + 3 tail rows per
//             layer at random places in a depth-sized region, x11 layers (23 MB)
//   gather1k  the same bytes as 2051 independent random 1 KiB rows per layer (worst case)
//   scan      the indexer's pooled-key read: 256 B every 3072 B (one pooled key per 4 rows of 768 B), depth/4 per
//             layer, x11 layers
// Usage: kvprobe <buffer GiB> [depths, comma list, tokens] [placements, comma list]
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <dirent.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static const int N_LAYER = 11;
static const size_t KV_ROW = 1024;       // 512 f16
static const size_t IDX_ROW = 768;       // 3 x 128 f16
static const size_t POOL_STRIDE = 4 * IDX_ROW;
static const size_t POOL_KEY = 256;      // 128 f16

static unsigned long cpu_node_mask() {
    unsigned long mask = 0;
    int nodes = 0;
    DIR * d = opendir("/sys/devices/system/node");
    if (!d) return 0;
    while (dirent * e = readdir(d)) {
        int n;
        if (sscanf(e->d_name, "node%d", &n) != 1 || n >= 64) continue;
        nodes++;
        std::string p = std::string("/sys/devices/system/node/") + e->d_name + "/cpulist";
        FILE * f = fopen(p.c_str(), "r");
        char buf[256] = {0};
        if (f) { if (!fgets(buf, sizeof(buf), f)) buf[0] = 0; fclose(f); }
        if (buf[0] >= '0' && buf[0] <= '9') mask |= 1ul << n;
    }
    closedir(d);
    return nodes > 1 ? mask : 0;
}

// one warp per 1 KiB unit: 32 lanes x 32 B
__global__ void k_gather(const uint8_t * __restrict__ src, const uint64_t * __restrict__ offs, int n_units, size_t unit, uint8_t * __restrict__ dst) {
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (w >= n_units) return;
    const uint4 * s = (const uint4 *) (src + offs[w]);
    uint4 * o = (uint4 *) (dst + (size_t) w * unit);
    const int n16 = (int) (unit / 16);
    for (int i = lane; i < n16; i += 32) o[i] = s[i];
}

// strided read: one 256 B key every `stride` bytes; reduce so the reads are not dead
__global__ void k_scan(const uint8_t * __restrict__ src, size_t n_keys, size_t stride, float * __restrict__ out) {
    const size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    const size_t key = t / 16;  // 16 lanes x 16 B per key
    const int lane = t % 16;
    float acc = 0.0f;
    if (key < n_keys) {
        uint4 v = ((const uint4 *) (src + key * stride))[lane];
        acc = __uint_as_float(v.x & 0x3f7fffff) + __uint_as_float(v.w & 0x3f7fffff);
    }
    if (acc == 12345.0f) out[0] = acc;
}

__global__ void k_fill(uint8_t * p, size_t n, uint8_t v) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) p[i] = v;
}

struct buf { std::string kind; uint8_t * p = nullptr; uint8_t * dp = nullptr; size_t size = 0; };

static bool alloc_buf(buf & b, const std::string & kind, size_t size) {
    b.kind = kind; b.size = size;
    if (kind == "dev") {
        if (cudaMalloc(&b.p, size) != cudaSuccess) { cudaGetLastError(); return false; }
        b.dp = b.p;
        return true;
    }
    if (kind == "halloc") {
        if (cudaHostAlloc(&b.p, size, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) { cudaGetLastError(); return false; }
        CK(cudaHostGetDevicePointer((void **) &b.dp, b.p, 0));
        return true;
    }
    // reg
    void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return false;
    const int thp = kind == "reg" ? madvise(p, size, MADV_HUGEPAGE) : -1;
    const unsigned long mask = cpu_node_mask();
    int mb = 0;
    if (mask) mb = (int) syscall(SYS_mbind, p, size, 2 /* MPOL_BIND */, &mask, 8 * sizeof(mask) + 1, 0);
    if (madvise(p, size, MADV_POPULATE_WRITE) != 0) {
        for (size_t o = 0; o < size; o += 4096) ((volatile char *) p)[o] = 0;
    }
    printf("  %s: MADV_HUGEPAGE=%d (-1 = not asked) cpu node mask=0x%lx mbind=%d\n", kind.c_str(), thp, mask, mb);
    if (cudaHostRegister(p, size, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
        printf("  cudaHostRegister failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        munmap(p, size);
        return false;
    }
    b.p = (uint8_t *) p;
    CK(cudaHostGetDevicePointer((void **) &b.dp, p, 0));
    return true;
}

static void free_buf(buf & b) {
    if (!b.p) return;
    if (b.kind == "dev") CK(cudaFree(b.p));
    else if (b.kind == "halloc") CK(cudaFreeHost(b.p));
    else { CK(cudaHostUnregister(b.p)); munmap(b.p, b.size); }
    b.p = b.dp = nullptr;
}

static bool check_ops(buf & b, uint8_t * dev_scratch) {
    bool ok = true;
    const size_t n = 1 << 20;
    std::vector<uint8_t> h(n);
    CK(cudaMemsetAsync(b.dp, 0x5a, n, 0)); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), b.dp, n, cudaMemcpyDeviceToHost));
    ok &= h[0] == 0x5a && h[n - 1] == 0x5a;
    printf("  memset via device ptr: %s\n", h[0] == 0x5a && h[n-1] == 0x5a ? "ok" : "FAIL");
    k_fill<<<64, 256>>>(dev_scratch, n, 0x33); CK(cudaDeviceSynchronize());
    CK(cudaMemcpyAsync(b.dp, dev_scratch, n, cudaMemcpyDeviceToDevice, 0)); CK(cudaDeviceSynchronize());
    bool c1 = b.p[0] == 0x33 && b.p[n - 1] == 0x33; ok &= c1;
    printf("  memcpy DeviceToDevice into it (host view): %s\n", c1 ? "ok" : "FAIL");
    std::fill(h.begin(), h.end(), 0x44);
    CK(cudaMemcpyAsync(b.dp, h.data(), n, cudaMemcpyHostToDevice, 0)); CK(cudaDeviceSynchronize());
    bool c2 = b.p[7] == 0x44; ok &= c2;
    printf("  memcpy HostToDevice into it: %s\n", c2 ? "ok" : "FAIL");
    k_fill<<<64, 256>>>(b.dp, n, 0x66); CK(cudaDeviceSynchronize());
    bool c3 = b.p[0] == 0x66 && b.p[n - 1] == 0x66; ok &= c3;
    printf("  kernel write, host reads it: %s\n", c3 ? "ok" : "FAIL");
    memset(b.p, 0x77, n);
    CK(cudaMemcpy(dev_scratch, b.dp, n, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(h.data(), dev_scratch, n, cudaMemcpyDeviceToHost));
    bool c4 = h[0] == 0x77 && h[n - 1] == 0x77; ok &= c4;
    printf("  host write, GPU copy sees it: %s\n", c4 ? "ok" : "FAIL");
    CK(cudaMemcpy(h.data(), b.dp, n, cudaMemcpyDefault));
    return ok;
}

int main(int argc, char ** argv) {
    const double gib = argc > 1 ? atof(argv[1]) : 4.0;
    std::string depths_s = argc > 2 ? argv[2] : "16384,100000";
    std::string kinds_s = argc > 3 ? argv[3] : "dev,reg,regnothp,halloc";
    const size_t size = (size_t) (gib * 1024 * 1024 * 1024);
    std::vector<long> depths; for (char * t = strtok(&depths_s[0], ","); t; t = strtok(nullptr, ",")) depths.push_back(atol(t));
    std::vector<std::string> kinds; { std::string s = kinds_s; size_t a = 0; while (a <= s.size()) { size_t c = s.find(',', a); if (c == std::string::npos) c = s.size(); kinds.push_back(s.substr(a, c - a)); a = c + 1; } }

    setvbuf(stdout, nullptr, _IOLBF, 0);
    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    int a1, a2, a3, a4, a5;
    cudaDeviceGetAttribute(&a1, cudaDevAttrCanUseHostPointerForRegisteredMem, dev);
    cudaDeviceGetAttribute(&a2, cudaDevAttrPageableMemoryAccess, dev);
    cudaDeviceGetAttribute(&a3, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev);
    cudaDeviceGetAttribute(&a4, cudaDevAttrHostNativeAtomicSupported, dev);
    cudaDeviceGetAttribute(&a5, cudaDevAttrIntegrated, dev);
    printf("device %s sm_%d%d, %d SMs, L2 %d MiB; canUseHostPtrForRegistered=%d pageableAccess=%d usesHostPageTables=%d nativeAtomics=%d integrated=%d; page %ld\n",
        prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize >> 20, a1, a2, a3, a4, a5, sysconf(_SC_PAGESIZE));

    const int units_max = N_LAYER * 2051 * 4;
    uint8_t * out; CK(cudaMalloc(&out, (size_t) units_max * 1024));
    uint64_t * d_offs; CK(cudaMalloc(&d_offs, (size_t) units_max * sizeof(uint64_t)));
    float * d_red; CK(cudaMalloc(&d_red, 64));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::mt19937_64 rng(1234);
    const int ITERS = 40;

    for (const auto & kind : kinds) {
        buf b;
        printf("== %s, %.1f GiB\n", kind.c_str(), gib);
        if (!alloc_buf(b, kind, size)) { printf("  alloc failed\n"); continue; }
        printf("  host %p device %p (%s)\n", (void *) b.p, (void *) b.dp, b.p == b.dp ? "same" : "different");
        if (kind != "dev") {
            bool ok = check_ops(b, out);
            printf("  ops %s\n", ok ? "all ok" : "SOME FAILED");
        }
        CK(cudaMemset(b.dp, 0, size));
        for (long depth : depths) {
            // the 11 layers' K for `depth` tokens laid out one after another, as llama_kv_cache does
            const size_t layer_kv = (size_t) depth * KV_ROW;
            const size_t layer_idx = (size_t) depth * IDX_ROW;
            if (N_LAYER * layer_kv > size || N_LAYER * layer_idx > size) { printf("  depth %ld does not fit\n", depth); continue; }
            for (int mode = 0; mode < 2; ++mode) {
                const int per_layer = mode == 0 ? 512 : 2051;
                double best = 1e9, sum = 0;
                for (int it = 0; it < ITERS; ++it) {
                    std::vector<uint64_t> offs;
                    for (int l = 0; l < N_LAYER; ++l) {
                        for (int u = 0; u < per_layer; ++u) {
                            const uint64_t row = mode == 0 ? (rng() % (depth / 4)) * 4 : rng() % depth;
                            offs.push_back(l * layer_kv + row * KV_ROW);
                        }
                    }
                    // 4 KiB units are gathered as four 1 KiB warps, the kernel's unit stays 1 KiB
                    std::vector<uint64_t> offs1;
                    if (mode == 0) { for (auto o : offs) for (int r = 0; r < 4; ++r) offs1.push_back(o + r * KV_ROW); }
                    else offs1 = offs;
                    if (mode == 0) for (int l = 0; l < N_LAYER; ++l) for (int r = 0; r < 3; ++r) offs1.push_back(l * layer_kv + (depth - 3 + r) * KV_ROW);
                    const int n = (int) offs1.size();
                    CK(cudaMemcpy(d_offs, offs1.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
                    CK(cudaEventRecord(e0));
                    k_gather<<<(n * 32 + 255) / 256, 256>>>(b.dp, d_offs, n, KV_ROW, out);
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                    if (it >= 3) { best = std::min(best, (double) ms); sum += ms; }
                }
                const double bytes = (mode == 0 ? (512.0 * 4 + 3) : 2051.0) * KV_ROW * N_LAYER;
                printf("  depth %7ld %-9s %7.1f us/token mean (best %6.1f), %6.1f GB/s\n", depth, mode == 0 ? "gather4k" : "gather1k",
                    1000.0 * sum / (ITERS - 3), 1000.0 * best, bytes / (sum / (ITERS - 3) / 1e3) / 1e9);
            }
            // indexer scan of all 11 layers' pooled keys
            {
                const size_t n_keys = (size_t) depth / 4;
                double sum = 0, best = 1e9;
                for (int it = 0; it < ITERS; ++it) {
                    CK(cudaEventRecord(e0));
                    for (int l = 0; l < N_LAYER; ++l) {
                        const size_t threads = n_keys * 16;
                        k_scan<<<(unsigned) ((threads + 255) / 256), 256>>>(b.dp + l * layer_idx + 2 * 256, n_keys, POOL_STRIDE, d_red);
                    }
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                    if (it >= 3) { best = std::min(best, (double) ms); sum += ms; }
                }
                const double bytes = (double) n_keys * POOL_KEY * N_LAYER;
                printf("  depth %7ld %-9s %7.1f us/token mean (best %6.1f), %6.1f GB/s of keys (%.1f MB)\n", depth, "scan",
                    1000.0 * sum / (ITERS - 3), 1000.0 * best, bytes / (sum / (ITERS - 3) / 1e3) / 1e9, bytes / 1e6);
            }
        }
        free_buf(b);
    }
    return 0;
}
