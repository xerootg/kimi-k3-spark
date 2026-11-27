// Cold-cache GEMV at production shapes with the weights in three placements:
//   dev    cudaMalloc (ggml CUDA buffer)
//   pinned cudaMallocHost (ggml CUDA host buffer type), read by the GPU through the UVA mapping
//   bank   anonymous mmap (+MADV_HUGEPAGE unless nothp) populated on the CPU, cudaHostRegister'ed through the fork's
//          host-bank buffer type: exactly how the expert store's bank is read in production
// usage: mmvq-bench2 <type> <k> <m> <dev|pinned|bank|banknothp> [total_MiB=1024] [reps=20]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <sys/mman.h>
#include <cuda_runtime.h>
#include <csignal>
#include <csetjmp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
extern "C" ggml_backend_buffer_t ggml_backend_cuda_host_bank_buffer_from_ptr(void * ptr, size_t size);
static ggml_type parse_type(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) { const char * n = ggml_type_name((ggml_type) t); if (n && strcasecmp(n, s) == 0) return (ggml_type) t; }
    fprintf(stderr, "unknown type %s\n", s); exit(1);
}
int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s type k m dev|pinned|bank|banknothp [total_MiB] [reps]\n", argv[0]); return 1; }
    const ggml_type type = parse_type(argv[1]);
    const int64_t k = atoll(argv[2]), m = atoll(argv[3]);
    const std::string mode = argv[4];
    const size_t total = (argc > 5 ? atoll(argv[5]) : 1024) << 20;
    const int reps = argc > 6 ? atoi(argv[6]) : 20;
    const size_t wbytes = ggml_row_size(type, k) * m;
    const int n = (int) std::max<size_t>(2, total / wbytes);
    ggml_backend_t be = ggml_backend_cuda_init(0);
    ggml_init_params ipw = { ggml_tensor_overhead() * (n + 2), nullptr, true };
    ggml_init_params ipg = { ggml_tensor_overhead() * (2 * n + 8) + ggml_graph_overhead_custom(4 * n, false), nullptr, true };
    ggml_context * ctxw = ggml_init(ipw);
    ggml_context * ctxg = ggml_init(ipg);
    std::vector<ggml_tensor *> w(n), y(n);
    for (int i = 0; i < n; i++) w[i] = ggml_new_tensor_2d(ctxw, type, k, m);
    ggml_backend_buffer_t bufw = nullptr; void * region = nullptr; size_t region_size = 0;
    if (mode == "dev") {
        bufw = ggml_backend_alloc_ctx_tensors(ctxw, be);
    } else if (mode == "pinned") {
        bufw = ggml_backend_alloc_ctx_tensors_from_buft(ctxw, ggml_backend_cuda_host_buffer_type());
    } else {
        const size_t align = 2u << 20;
        region_size = ((size_t) n * ((wbytes + 511) & ~(size_t) 511) + align - 1) & ~(align - 1);
        region = mmap(nullptr, region_size + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region == MAP_FAILED) { perror("mmap"); return 1; }
        region = (void *) (((uintptr_t) region + align - 1) & ~(uintptr_t) (align - 1));
        if (mode == "bank") madvise(region, region_size, MADV_HUGEPAGE);
        memset(region, 0, region_size); // populate on the CPU, as the store does
        bufw = ggml_backend_cuda_host_bank_buffer_from_ptr(region, region_size);
        if (!bufw) { fprintf(stderr, "bank buffer failed\n"); return 1; }
        ggml_tallocr ta = ggml_tallocr_new(bufw);
        for (int i = 0; i < n; i++) ggml_tallocr_alloc(&ta, w[i]);
    }
    if (mode == "managed" || mode == "managedpref" || mode == "devptr") {
        // allocate as a CUDA buffer for the backend's buffer checks, then point every tensor at the test memory
        bufw = ggml_backend_alloc_ctx_tensors(ctxw, be);
        const size_t stride = (wbytes + 511) & ~(size_t) 511;
        void * p = nullptr;
        if (mode == "devptr") { if (cudaMalloc(&p, stride * n) != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); return 1; } }
        else if (cudaMallocManaged(&p, stride * n, cudaMemAttachGlobal) != cudaSuccess) { fprintf(stderr, "cudaMallocManaged failed\n"); return 1; }
        if (mode == "managedpref") {
            cudaMemLocation dl{}; dl.type = cudaMemLocationTypeDevice; dl.id = 0;
            cudaMemLocation hl{}; hl.type = cudaMemLocationTypeHost; hl.id = 0;
            cudaMemAdvise(p, stride * n, cudaMemAdviseSetPreferredLocation, dl);
            cudaMemAdvise(p, stride * n, cudaMemAdviseSetAccessedBy, hl);
        }
        for (int i = 0; i < n; i++) w[i]->data = (char *) p + stride * i;
        region = p;
    }
    ggml_tensor * x = ggml_new_tensor_1d(ctxg, GGML_TYPE_F32, k);
    ggml_cgraph * gf = ggml_new_graph_custom(ctxg, 4 * n, false);
    for (int i = 0; i < n; i++) { y[i] = ggml_mul_mat(ctxg, w[i], x); ggml_build_forward_expand(gf, y[i]); }
    ggml_backend_buffer_t bufg = ggml_backend_alloc_ctx_tensors(ctxg, be);
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> f(k * m); for (auto & v : f) v = nd(rng);
    std::vector<uint8_t> q(wbytes);
    ggml_quantize_chunk(type, f.data(), q.data(), 0, m, k, nullptr);
    std::vector<float> xv(k); for (auto & v : xv) v = nd(rng);
    ggml_backend_tensor_set(x, xv.data(), 0, k * sizeof(float));
    const size_t rb = ggml_row_size(type, k);
    std::vector<uint8_t> qi(wbytes);
    for (int i = 0; i < n; i++) {
        const int64_t sh = (i * 7) % m;
        memcpy(qi.data(), q.data() + sh * rb, (m - sh) * rb);
        memcpy(qi.data() + (m - sh) * rb, q.data(), sh * rb);
        if (mode == "dev" || mode == "pinned" || mode == "devptr") { if (mode == "devptr") cudaMemcpy(w[i]->data, qi.data(), wbytes, cudaMemcpyHostToDevice); else ggml_backend_tensor_set(w[i], qi.data(), 0, wbytes); }
        else memcpy(w[i]->data, qi.data(), wbytes);
    }
    ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
    ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
    double best = 1e30, sum = 0;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
        best = std::min(best, us); sum += us;
    }
    if (mode == "managed" || mode == "managedpref" || mode == "devptr") {
        // can the CPU write this memory after the GPU used it? (devptr: expect a fault on a dGPU; GB10 is the question)
        static sigjmp_buf jb; struct sigaction sa{}; sa.sa_handler = [](int) { siglongjmp(jb, 1); }; sigaction(SIGSEGV, &sa, nullptr); sigaction(SIGBUS, &sa, nullptr);
        if (sigsetjmp(jb, 1) == 0) { auto t0 = std::chrono::steady_clock::now(); memset(w[0]->data, 0, wbytes); double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); printf("cpu write ok: %.1f MB in %.0f us (%.1f GB/s)\n", wbytes/1e6, us, wbytes/(us*1e3)); }
        else printf("cpu write FAULTS\n");
        signal(SIGSEGV, SIG_DFL); signal(SIGBUS, SIG_DFL);
    }
    uint64_t h = 1469598103934665603ull; std::vector<uint8_t> out(m * sizeof(float));
    for (int i = 0; i < n; i++) { ggml_backend_tensor_get(y[i], out.data(), 0, out.size()); for (uint8_t c : out) { h ^= c; h *= 1099511628211ull; } }
    printf("%-8s %-9s k=%-6lld m=%-6lld copies=%-4d %.2f MB  best %8.2f us  mean %8.2f us  %6.1f GB/s  digest %016llx\n",
           ggml_type_name(type), mode.c_str(), (long long) k, (long long) m, n, wbytes / 1e6, best, sum / reps, wbytes / (best * 1e3), (unsigned long long) h);
    return 0;
}
