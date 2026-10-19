// uvmbw.cu: can a GH200 GPU read MoE experts out of host memory through the unified-memory paths (ATS on plain malloc,
// THP, cudaMallocManaged with hints, mmap of the expert file) as fast as through cudaHostRegister?
// Pattern: each step reads 16 chunks of 6.6 MB (one K3 MoE layer) at random 64 KiB-aligned offsets in a large region.
// "cold" = first step on fresh offsets (translations not cached), "warm" = the same offsets read again.
// usage: uvmbw MODE REGION_GIB [FILE]
//   MODE: reg | ats | thp | managed | managed_hint | managed_hbm | hbm_ats | hbm_reg | file | file_cold | file_reg | file_hbm | file_pf | file_pf_hbm
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <atomic>
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s failed: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void chunk_kernel(const char * base, const size_t * offs, size_t chunk, int blocks_per_chunk, unsigned long long * out) {
    const int c = blockIdx.x / blocks_per_chunk, b = blockIdx.x % blocks_per_chunk;
    const int4 * p = (const int4 *) (base + offs[c]);
    const size_t n = chunk / 16, per = (n + blocks_per_chunk - 1) / blocks_per_chunk;
    size_t i = b * per + threadIdx.x, end = (b + 1) * per; if (end > n) end = n;
    unsigned long long acc = 0;
    for (; i < end; i += blockDim.x) { int4 v = p[i]; acc += (unsigned) v.x + (unsigned) v.y; }
    if (acc == 0xdeadbeefULL) *out = acc;
}

// memory policy without libnuma: MPOL_BIND = 2
static long set_policy_node(int node) { unsigned long mask = 1ul << node; return syscall(SYS_set_mempolicy, 2, &mask, 64); }
static long mbind_node(void * p, size_t len, int node) { unsigned long mask = 1ul << node; return syscall(SYS_mbind, p, len, 2, &mask, 64, 0); }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void smaps_huge() {
    FILE * f = fopen("/proc/self/smaps_rollup", "r"); char line[256];
    while (f && fgets(line, sizeof line, f)) if (strstr(line, "AnonHugePages") || strstr(line, "FilePmdMapped") || strstr(line, "Rss:")) printf("    %s", line);
    if (f) fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 3) { printf("usage: uvmbw MODE REGION_GIB [FILE]\n"); return 1; }
    const std::string mode = argv[1];
    const size_t GiB = 1ull << 30, region = (size_t) atoll(argv[2]) * GiB, chunk = 6619136, align = 64 << 10;
    const int n_chunks = 16, bpc = 64, steps = 12;
    unsigned long long * out; CK(cudaMalloc(&out, 8));
    size_t * d_offs; CK(cudaMalloc(&d_offs, n_chunks * sizeof(size_t)));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    char * base = nullptr; double t0 = now();
    if (mode == "hbm_ats" || mode == "hbm_reg") {  // ordinary host allocation, physically placed on the HBM NUMA node
        base = (char *) aligned_alloc(512ull << 20, region);
        if (mbind_node(base, region, 1) != 0) perror("mbind node 1");
        memset(base, 1, region);
        if (mode == "hbm_reg") CK(cudaHostRegister(base, region, cudaHostRegisterPortable));
    } else if (mode == "reg" || mode == "ats" || mode == "thp") {
        base = (char *) aligned_alloc(512ull << 20, region);
        if (mode == "thp" && madvise(base, region, MADV_HUGEPAGE) != 0) perror("madvise HUGEPAGE");
        memset(base, 1, region);
        if (mode == "reg") CK(cudaHostRegister(base, region, cudaHostRegisterPortable));
    } else if (mode.rfind("managed", 0) == 0) {
        CK(cudaMallocManaged(&base, region));
        if (mode != "managed") {
            CK(cudaMemAdvise(base, region, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
            CK(cudaMemAdvise(base, region, cudaMemAdviseSetAccessedBy, 0));
        }
        memset(base, 1, region);
    } else if (mode.rfind("file", 0) == 0) {
        if (argc < 4) { printf("file mode needs FILE\n"); return 1; }
        int fd = open(argv[3], O_RDONLY); if (fd < 0) { perror("open"); return 1; }
        if (mode != "file" && mode != "file_reg") posix_fadvise(fd, 0, region, POSIX_FADV_DONTNEED);
        if (mode == "file_hbm") set_policy_node(1);  // page cache filled by this thread lands on the HBM node
        base = (char *) mmap(nullptr, region, PROT_READ, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { perror("mmap"); return 1; }
        if (mode == "file" || mode == "file_reg" || mode == "file_hbm") {  // page-cache warm: CPU reads the whole region once
            volatile unsigned long long s = 0; for (size_t i = 0; i < region; i += 4096) s += base[i];
        }
        if (mode == "file_hbm") set_policy_node(0);
        if (mode == "file_reg") CK(cudaHostRegister(base, region, cudaHostRegisterReadOnly | cudaHostRegisterPortable));
    } else { printf("unknown mode\n"); return 1; }
    printf("%s: region %zu GiB set up in %.1f s\n", mode.c_str(), region / GiB, now() - t0);
    smaps_huge();

    std::vector<size_t> offs(n_chunks);
    srand(1234);
    double cold_sum = 0, warm_sum = 0; int n = 0;
    for (int step = 0; step < steps; ++step) {
        for (auto & o : offs) o = ((size_t) rand() * (size_t) rand() % ((region - chunk) / align)) * align;
        CK(cudaMemcpy(d_offs, offs.data(), n_chunks * sizeof(size_t), cudaMemcpyHostToDevice));
        if (mode == "managed_hbm") {  // the tiering mechanic: migrate this step's experts to HBM, then read them there
            double tm = now();
            for (auto o : offs) CK(cudaMemPrefetchAsync(base + o, chunk, 0));
            CK(cudaDeviceSynchronize());
            if (step >= 2) printf("    migrate 16 chunks to HBM: %.3f ms = %.1f GB/s\n", (now() - tm) * 1e3, n_chunks * chunk / 1e9 / (now() - tm));
        }
        double t_pf = 0;
        if (mode == "file_pf" || mode == "file_pf_hbm") {
            double tp = now();
            const size_t piece = 1 << 20; const size_t n_pieces = chunk / piece + 1;
            std::atomic<size_t> next{0}; std::vector<std::thread> th;
            for (int t = 0; t < 32; ++t) th.emplace_back([&] {
                if (mode == "file_pf_hbm") set_policy_node(1);
                for (size_t k; (k = next++) < n_chunks * n_pieces; ) {
                    const size_t c = k / n_pieces, off = (k % n_pieces) * piece; if (off >= chunk) continue;
                    const size_t len = std::min(piece, chunk - off);
                    char * a = base + offs[c] + off; char * al = (char *) ((uintptr_t) a & ~(uintptr_t) (align - 1));
                    if (madvise(al, len + (a - al), MADV_POPULATE_READ) != 0) { perror("populate"); exit(1); }
                }
            });
            for (auto & t : th) t.join();
            t_pf = now() - tp;
            if (step >= 2 && step < 5) printf("    populate 16 chunks: %.2f ms = %.1f GB/s\n", t_pf * 1e3, n_chunks * chunk / 1e9 / t_pf);
        }
        float ms[2];
        for (int pass = 0; pass < 2; ++pass) {
            CK(cudaEventRecord(e0));
            chunk_kernel<<<n_chunks * bpc, 256>>>(base, d_offs, chunk, bpc, out);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            CK(cudaEventElapsedTime(&ms[pass], e0, e1));
        }
        if (step >= 2) { cold_sum += ms[0]; warm_sum += ms[1]; ++n; }
    }
    const double gb = n_chunks * chunk / 1e9;
    printf("%-14s cold %8.1f GB/s (%.3f ms/layer)   warm %8.1f GB/s (%.3f ms/layer)\n", mode.c_str(),
           gb / (cold_sum / n / 1e3), cold_sum / n, gb / (warm_sum / n / 1e3), warm_sum / n);
    return 0;
}
