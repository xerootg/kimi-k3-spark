// Cold-cache GEMV bench at production shapes: N distinct weight copies (> L2) multiplied by one
// vector in a single graph, like decode reads each weight once per token. Prints us per matrix,
// GB/s, and an FNV digest of every output so kernel changes can be checked for identical bytes.
// usage: mmvq-bench <type> <k> <m> [total_MiB=1024] [reps=20]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_type parse_type(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && strcasecmp(n, s) == 0) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", s); exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s type k m [total_MiB] [reps]\n", argv[0]); return 1; }
    const ggml_type type = parse_type(argv[1]);
    const int64_t k = atoll(argv[2]), m = atoll(argv[3]);
    const size_t total = (argc > 4 ? atoll(argv[4]) : 1024) << 20;
    const int reps = argc > 5 ? atoi(argv[5]) : 20;
    const size_t wbytes = ggml_row_size(type, k) * m;
    const int n = (int) std::max<size_t>(2, total / wbytes);

    ggml_backend_t be = ggml_backend_cuda_init(0);
    ggml_init_params ip = { ggml_tensor_overhead() * (3 * n + 8) + ggml_graph_overhead_custom(4 * n, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    std::vector<ggml_tensor *> w(n), y(n);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4 * n, false);
    for (int i = 0; i < n; i++) {
        w[i] = ggml_new_tensor_2d(ctx, type, k, m);
        y[i] = ggml_mul_mat(ctx, w[i], x);
        ggml_build_forward_expand(gf, y[i]);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    // one valid quantized matrix (from N(0,1) floats), copied to every slot with a per-copy row rotation
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
        ggml_backend_tensor_set(w[i], qi.data(), 0, wbytes);
    }

    ggml_backend_graph_compute(be, gf);  // warm-up (and CUDA graph capture)
    ggml_backend_graph_compute(be, gf);
    double best = 1e30, sum = 0;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
        best = std::min(best, us); sum += us;
    }
    uint64_t h = 1469598103934665603ull;
    std::vector<uint8_t> out(m * sizeof(float));
    for (int i = 0; i < n; i++) {
        ggml_backend_tensor_get(y[i], out.data(), 0, out.size());
        for (uint8_t c : out) { h ^= c; h *= 1099511628211ull; }
    }
    printf("%-8s k=%-6lld m=%-6lld copies=%-4d %.2f MB  best %8.2f us  mean %8.2f us  %6.1f GB/s  digest %016llx\n",
           ggml_type_name(type), (long long) k, (long long) m, n, wbytes / 1e6, best, sum / reps,
           wbytes / (best * 1e3), (unsigned long long) h);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
    return 0;
}
