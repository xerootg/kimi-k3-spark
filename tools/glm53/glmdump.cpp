// glmdump: reference dumps of GLM-5.3's routed-expert FFN for one decode token, computed by the production
// ggml-cuda kernels (link against the production tree's libggml-base and load its libggml-cuda.so).
//
// Graph: build_moe_ffn LLM_FFN_SILU, no biases, scales or clamp.
// Compile on Spark:
// g++ -O2 -std=c++17 -I ~/llama-k3-sc/ggml/include tools/glm53/glmdump.cpp -o glmdump
// (continue command with) -L ~/llama-k3-sc/build/bin -Wl,-rpath,$HOME/llama-k3-sc/build/bin -lggml -lggml-base
//
// usage: glmdump CUDA_LIB RECORDS_FILE N_RECORDS GATE_TYPE UP_TYPE DOWN_TYPE RECORD_SIZE N_CASES SEED OUTDIR
//   records: gate | up | down back to back (store layout), record size = sum of the three, unpadded or padded
//   writes OUTDIR/{x,ids,g,u,h,dn}.bin; every case uses 8 ids (a permutation of a random subset of the records)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

static const int D = 6144, F = 2048, K = 8;

int main(int argc, char ** argv) {
    if (argc < 11) { fprintf(stderr, "usage: glmdump CUDA_LIB RECORDS N_RECORDS GATE_T UP_T DOWN_T RECORD_SIZE N_CASES SEED OUTDIR\n"); return 1; }
    const char * lib = argv[1];
    const char * path = argv[2];
    const int E = atoi(argv[3]);
    const ggml_type tg = (ggml_type) atoi(argv[4]), tu = (ggml_type) atoi(argv[5]), td = (ggml_type) atoi(argv[6]);
    const size_t recsz = strtoull(argv[7], nullptr, 10);
    const int ncase = atoi(argv[8]);
    const unsigned seed = (unsigned) atoi(argv[9]);
    const std::string out = argv[10];
    std::filesystem::create_directories(out);

    if (E < K || ncase <= 0) { fprintf(stderr,"need >=8 records and positive cases\n"); return 1; }
    ggml_backend_reg_t reg = ggml_backend_load(lib);
    if (!reg) { fprintf(stderr, "cannot load %s\n", lib); return 1; }
    if (ggml_backend_reg_dev_count(reg) == 0) { fprintf(stderr,"no CUDA device\n"); return 1; }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    fprintf(stderr, "device: %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    const size_t gsz = ggml_row_size(tg, D) * F, usz = ggml_row_size(tu, D) * F, dsz = ggml_row_size(td, F) * D;
    if (gsz + usz + dsz > recsz) { fprintf(stderr, "record size too small\n"); return 1; }

    ggml_init_params wp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * wg = ggml_new_tensor_3d(wctx, tg, D, F, E);
    ggml_tensor * wu = ggml_new_tensor_3d(wctx, tu, D, F, E);
    ggml_tensor * wd = ggml_new_tensor_3d(wctx, td, F, D, E);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    {
        FILE * f = fopen(path, "rb");
        if (!f) { perror(path); return 1; }
        std::vector<char> buf(recsz);
        for (int e = 0; e < E; e++) {
            if (fread(buf.data(), 1, recsz, f) != recsz) { fprintf(stderr, "short read %d\n", e); return 1; }
            ggml_backend_tensor_set(wg, buf.data(), e * gsz, gsz);
            ggml_backend_tensor_set(wu, buf.data() + gsz, e * usz, usz);
            ggml_backend_tensor_set(wd, buf.data() + gsz + usz, e * dsz, dsz);
        }
        fclose(f);
    }

    // one graph per output so that no intermediate is marked as an output (that would stop the fusions):
    // 0: gate rows, 1: up rows, 2: h (activation fed to down), 3: down rows
    struct G { ggml_context * ctx; ggml_cgraph * gf; ggml_tensor * x, * ids, * out; ggml_gallocr_t ga; };
    G gr[4];
    for (int v = 0; v < 4; v++) {
        ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        G & g = gr[v];
        g.ctx = ggml_init(gp);
        ggml_context * ctx = g.ctx;
        g.x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
        g.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, 1);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, wg, g.x, g.ids);           // [F, K, 1]
        ggml_tensor * up   = ggml_mul_mat_id(ctx, wu, g.x, g.ids);
        // llama-graph.cpp:2371-2425: separate up/gate + split SwiGLU.
        ggml_tensor * h = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * dn = ggml_mul_mat_id(ctx, wd, h, g.ids);               // [D, K, 1]
        g.out = v == 0 ? gate : v == 1 ? up : v == 2 ? h : dn;
        ggml_set_output(g.out);
        g.gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(g.gf, g.out);
        g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(g.ga, g.gf)) { fprintf(stderr,"graph allocation failed\n"); return 1; }
    }

    std::mt19937 rng(seed);
    std::vector<float> x(D);
    std::vector<int> ids(K), perm(E);
    std::vector<float> o(D * K);
    const char * names[4] = {"g", "u", "h", "dn"};
    const int width[4] = {F, F, F, D};
    FILE * fo[4]; for (int v = 0; v < 4; v++) fo[v] = fopen((out + "/" + names[v] + ".bin").c_str(), "wb");
    FILE * fx = fopen((out + "/x.bin").c_str(), "wb");
    FILE * fi = fopen((out + "/ids.bin").c_str(), "wb");
    FILE * fk = fopen((out + "/kind.txt").c_str(), "w");

    for (FILE * f : fo) if (!f) { perror(out.c_str()); return 1; }
    if (!fx || !fi || !fk) { perror(out.c_str()); return 1; }

    for (int c = 0; c < ncase; c++) {
        // activation: mostly random normal at several scales, some edge cases
        std::normal_distribution<float> nd(0.f, 1.f);
        std::uniform_real_distribution<float> ud(0.f, 1.f);
        const int kind = c < 8 ? c : 8 + (int) (rng() % 6);
        float sc = 1.0f;
        std::string desc;
        switch (kind) {
            case 0: for (auto & v : x) v = 0.0f; desc = "zeros"; break;
            case 1: for (auto & v : x) v = 0.0f; x[rng() % D] = 3.0f; x[rng() % D] = -0.5f; desc = "two spikes"; break;
            case 2: for (auto & v : x) v = nd(rng) * 1e-30f; desc = "tiny 1e-30 (fp32 normal)"; break;
            case 3: for (auto & v : x) v = nd(rng) * 1e-39f; desc = "denormals"; break;
            case 4: for (auto & v : x) v = nd(rng) * 30.0f; desc = "large sigma 30"; break;
            case 5: for (int i = 0; i < D; i++) x[i] = (i / 32) % 3 == 0 ? 0.0f : nd(rng); desc = "zero blocks"; break;
            case 6: for (auto & v : x) v = 0.25f; desc = "constant 0.25"; break;
            case 7: for (auto & v : x) v = nd(rng) * 1000.0f; desc = "huge sigma 1000"; break;
            default: {
                const float scales[6] = {0.02f, 0.1f, 0.3f, 1.0f, 3.0f, 10.0f};
                sc = scales[kind - 8];
                for (auto & v : x) v = nd(rng) * sc;
                // heavy tail on a few channels, like real hidden states
                for (int t = 0; t < 8; t++) x[rng() % D] *= 1.0f + 20.0f * ud(rng);
                desc = "normal sigma " + std::to_string(sc) + " + 8 outliers";
            }
        }
        for (int i = 0; i < E; i++) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < K; i++) ids[i] = perm[i];
        fwrite(x.data(), 4, D, fx);
        fwrite(ids.data(), 4, K, fi);
        fprintf(fk, "%d %s\n", c, desc.c_str());
        for (int v = 0; v < 4; v++) {
            ggml_backend_tensor_set(gr[v].x, x.data(), 0, D * 4);
            ggml_backend_tensor_set(gr[v].ids, ids.data(), 0, K * 4);
            if (ggml_backend_graph_compute(be, gr[v].gf) != GGML_STATUS_SUCCESS) return 2;
            ggml_backend_tensor_get(gr[v].out, o.data(), 0, (size_t) width[v] * K * 4);
            fwrite(o.data(), 4, (size_t) width[v] * K, fo[v]);
        }
        // determinism check on the last output: recompute once and compare bits
        if (c % 50 == 0) {
            std::vector<float> o2(D * K);
            if (ggml_backend_graph_compute(be, gr[3].gf) != GGML_STATUS_SUCCESS) return 2;
            ggml_backend_tensor_get(gr[3].out, o2.data(), 0, (size_t) D * K * 4);
            if (memcmp(o.data(), o2.data(), (size_t) D * K * 4)) { fprintf(stderr, "case %d: NOT deterministic\n", c); return 2; }
        }
    }
    for (int v = 0; v < 4; v++) fclose(fo[v]);
    fclose(fx); fclose(fi); fclose(fk);
    fprintf(stderr, "wrote %d cases\n", ncase);
    for (auto & g : gr) { ggml_gallocr_free(g.ga); ggml_free(g.ctx); }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(be);
    return 0;
}
