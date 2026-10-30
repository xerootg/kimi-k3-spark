// glmbench: one GLM-5.3 MoE layer's routed-expert FFN for one token (MUL_MAT_ID gate, up, swiglu, down) on one ggml
// backend, from real expert slices (tools/glm53/extract.py: PREFIX_{gate,up,down}.bin, 8 experts each).
//
// usage: glmbench BACKEND PREFIX TG TU TD "k1,k2,..." ITERS [THREADS] [--check OUT]
//   types are ggml ids (IQ2_XS 17, IQ3_XXS 18, IQ4_XS 23, Q4_K 12, Q5_K 13, Q6_K 14)
//   The 8 real experts are replicated into SLOTS (64) slots so a token's weights stream from DRAM, not cache.
//   --check: ids 0..7, writes OUT.x (6144 f32) and OUT.dn (8 x 6144 f32, unweighted down rows) for check.py.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int D = 6144, F = 2048, NE = 8, SLOTS = 64;

static std::vector<char> slurp(const std::string & p, size_t n) {
    std::vector<char> b(n);
    FILE * f = fopen(p.c_str(), "rb");
    if (!f || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "read %s failed\n", p.c_str()); exit(1); }
    fclose(f);
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: see source\n"); return 1; }
    std::string bname = argv[1], pre = argv[2];
    ggml_type tg = (ggml_type) atoi(argv[3]), tu = (ggml_type) atoi(argv[4]), td = (ggml_type) atoi(argv[5]);
    std::vector<int> ks;
    for (char * s = strtok(argv[6], ","); s; s = strtok(nullptr, ",")) ks.push_back(atoi(s));
    int iters = atoi(argv[7]);
    int threads = argc > 8 ? atoi(argv[8]) : 8;
    const char * check = (argc > 10 && !strcmp(argv[9], "--check")) ? argv[10] : nullptr;
    int slots = check ? NE : SLOTS;

    ggml_backend_load_all();
    ggml_backend_t be = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev = ggml_backend_dev_get(i);
        fprintf(stderr, "device %zu: %s (%s)\n", i, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        if (bname == ggml_backend_dev_name(dev)) be = ggml_backend_dev_init(dev, nullptr);
    }
    if (!be) { fprintf(stderr, "backend %s not found\n", bname.c_str()); return 1; }
    if (auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(be))) {
        auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn) fn(be, threads);
    }

    size_t gsz = ggml_row_size(tg, D) * F, usz = ggml_row_size(tu, D) * F, dsz = ggml_row_size(td, F) * D;
    size_t rec = gsz + usz + dsz;
    fprintf(stderr, "expert %zu bytes (gate %zu, up %zu, down %zu), %d slots = %.1f MB\n", rec, gsz, usz, dsz, slots, rec * slots / 1e6);
    auto bg = slurp(pre + "_gate.bin", gsz * NE), bu = slurp(pre + "_up.bin", usz * NE), bd = slurp(pre + "_down.bin", dsz * NE);

    ggml_init_params wp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * wg = ggml_new_tensor_3d(wctx, tg, D, F, slots);
    ggml_tensor * wu = ggml_new_tensor_3d(wctx, tu, D, F, slots);
    ggml_tensor * wd = ggml_new_tensor_3d(wctx, td, F, D, slots);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int s = 0; s < slots; s++) {
        int e = s % NE;
        ggml_backend_tensor_set(wg, bg.data() + e * gsz, s * gsz, gsz);
        ggml_backend_tensor_set(wu, bu.data() + e * usz, s * usz, usz);
        ggml_backend_tensor_set(wd, bd.data() + e * dsz, s * dsz, dsz);
    }

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xv(D);
    for (auto & v : xv) v = nd(rng);

    for (int k : ks) {
        ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, 1);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, 1);
        ggml_tensor * g = ggml_mul_mat_id(ctx, wg, x, ids);           // [F, k, 1]
        ggml_tensor * u = ggml_mul_mat_id(ctx, wu, x, ids);
        ggml_tensor * a = ggml_swiglu_split(ctx, g, u);
        ggml_tensor * dn = ggml_mul_mat_id(ctx, wd, a, ids);          // [D, k, 1]
        ggml_tensor * y = ggml_mul(ctx, dn, w);
        ggml_tensor * out = ggml_view_2d(ctx, y, D, 1, y->nb[1], 0);
        for (int j = 1; j < k; j++) out = ggml_add(ctx, out, ggml_view_2d(ctx, y, D, 1, y->nb[1], j * y->nb[1]));
        if (check) { ggml_set_output(g); ggml_set_output(a); ggml_set_output(dn); }   // keep gallocr from reusing them in place
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(x, xv.data(), 0, D * sizeof(float));
        std::vector<float> wv(k, 1.0f / k);
        ggml_backend_tensor_set(w, wv.data(), 0, k * sizeof(float));

        if (check) {
            std::vector<int> id0(k); for (int i = 0; i < k; i++) id0[i] = i;
            ggml_backend_tensor_set(ids, id0.data(), 0, k * sizeof(int));
            ggml_backend_graph_compute(be, gf);
            std::vector<float> rows((size_t) k * D);
            ggml_backend_tensor_get(dn, rows.data(), 0, rows.size() * sizeof(float));
            std::string o = check;
            FILE * f = fopen((o + ".x").c_str(), "wb"); fwrite(xv.data(), 4, D, f); fclose(f);
            f = fopen((o + ".dn").c_str(), "wb"); fwrite(rows.data(), 4, rows.size(), f); fclose(f);
            std::vector<float> gr((size_t) k * F);
            ggml_backend_tensor_get(g, gr.data(), 0, gr.size() * 4);
            f = fopen((o + ".g").c_str(), "wb"); fwrite(gr.data(), 4, gr.size(), f); fclose(f);
            ggml_backend_tensor_get(a, gr.data(), 0, gr.size() * 4);
            f = fopen((o + ".h").c_str(), "wb"); fwrite(gr.data(), 4, gr.size(), f); fclose(f);
            fprintf(stderr, "wrote %s.{x,g,h,dn}\n", check);
        } else {
            std::vector<int> perm(slots);
            std::vector<double> t;
            for (int it = 0; it < iters + 5; it++) {
                for (int i = 0; i < slots; i++) perm[i] = i;
                std::shuffle(perm.begin(), perm.end(), rng);
                ggml_backend_tensor_set(ids, perm.data(), 0, k * sizeof(int));
                auto t0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(be, gf);
                ggml_backend_synchronize(be);
                auto t1 = std::chrono::steady_clock::now();
                if (it >= 5) t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(t.begin(), t.end());
            double med = t[t.size() / 2], p90 = t[(size_t) (t.size() * 0.9)];
            printf("%s %s k=%d thr=%d: p50 %.0f us p90 %.0f us  %.0f us/expert  %.1f GB/s\n",
                   bname.c_str(), pre.substr(pre.rfind('/') + 1).c_str(), k, threads, med, p90, med / k, k * rec / med / 1e3);
            fflush(stdout);
        }
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(be);
    return 0;
}
