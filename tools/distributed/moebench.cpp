// moebench: one Kimi K3 layer's routed-expert FFN for a single token on one ggml backend, from real
// expert-store records (gate | up | down back to back), to time remote expert compute.
//
// usage: moebench BACKEND RECORDS_FILE N_RECORDS GATE_TYPE DOWN_TYPE "k1,k2,..." ITERS [THREADS] [--dump OUT]
//   BACKEND: CPU or e.g. SYCL0 (name as ggml lists it)
//   types: ggml type ids (19 = IQ1_S, 16 = IQ2_XXS)
// Per iteration it draws k distinct experts out of N_RECORDS so the weights stream from DRAM.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

static const int D = 3584, F = 3072;

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: see source\n"); return 1; }
    std::string bname = argv[1];
    const char * path = argv[2];
    int E = atoi(argv[3]);
    ggml_type tg = (ggml_type) atoi(argv[4]), td = (ggml_type) atoi(argv[5]);
    std::vector<int> ks;
    int serve_port = 0;
    if (!strncmp(argv[6], "serve:", 6)) { serve_port = atoi(argv[6] + 6); for (int k = 1; k <= 16; k++) ks.push_back(k); }
    else for (char * s = strtok(argv[6], ","); s; s = strtok(nullptr, ",")) ks.push_back(atoi(s));
    int iters = atoi(argv[7]);
    int threads = argc > 8 ? atoi(argv[8]) : 8;

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

    size_t gsz = ggml_row_size(tg, D) * F, dsz = ggml_row_size(td, F) * D;
    size_t rec = gsz * 2 + dsz;
    fprintf(stderr, "record %zu bytes (gate/up %zu, down %zu), %d records = %.1f MB\n", rec, gsz, dsz, E, rec * E / 1e6);

    ggml_init_params wp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * wg = ggml_new_tensor_3d(wctx, tg, D, F, E);
    ggml_tensor * wu = ggml_new_tensor_3d(wctx, tg, D, F, E);
    ggml_tensor * wd = ggml_new_tensor_3d(wctx, td, F, D, E);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    std::vector<char> buf(rec);
    for (int e = 0; e < E; e++) {
        if (fread(buf.data(), 1, rec, f) != rec) { fprintf(stderr, "short read at record %d\n", e); return 1; }
        ggml_backend_tensor_set(wg, buf.data(), e * gsz, gsz);
        ggml_backend_tensor_set(wu, buf.data() + gsz, e * gsz, gsz);
        ggml_backend_tensor_set(wd, buf.data() + 2 * gsz, e * dsz, dsz);
    }
    fclose(f);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xv(D);
    for (auto & v : xv) v = nd(rng);

    struct G { ggml_context * ctx; ggml_cgraph * gf; ggml_tensor * x, * ids, * dn; ggml_gallocr_t ga; };
    std::vector<G> graphs(17);
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
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(x, xv.data(), 0, D * sizeof(float));
        std::vector<float> wv(k, 1.0f / k);
        ggml_backend_tensor_set(w, wv.data(), 0, k * sizeof(float));

        if (serve_port) {
            ggml_gallocr_free(ga);
            ggml_free(ctx);
            ggml_init_params gp2 = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            G & gr = graphs[k];
            gr.ctx = ggml_init(gp2);
            gr.x = ggml_new_tensor_3d(gr.ctx, GGML_TYPE_F32, D, 1, 1);
            gr.ids = ggml_new_tensor_2d(gr.ctx, GGML_TYPE_I32, k, 1);
            ggml_tensor * g2 = ggml_mul_mat_id(gr.ctx, wg, gr.x, gr.ids);
            ggml_tensor * u2 = ggml_mul_mat_id(gr.ctx, wu, gr.x, gr.ids);
            gr.dn = ggml_mul_mat_id(gr.ctx, wd, ggml_swiglu_split(gr.ctx, g2, u2), gr.ids);   // [D, k, 1] rows, unweighted
            gr.gf = ggml_new_graph(gr.ctx);
            ggml_build_forward_expand(gr.gf, gr.dn);
            gr.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            ggml_gallocr_alloc_graph(gr.ga, gr.gf);
            std::vector<int> id0(k); for (int i = 0; i < k; i++) id0[i] = i;
            ggml_backend_tensor_set(gr.ids, id0.data(), 0, k * sizeof(int));
            ggml_backend_tensor_set(gr.x, xv.data(), 0, D * sizeof(float));
            ggml_backend_graph_compute(be, gr.gf);   // warm-up / JIT
            continue;
        }
        std::vector<int> perm(E);
        std::vector<double> t;
        for (int it = 0; it < iters + 5; it++) {
            for (int i = 0; i < E; i++) perm[i] = i;
            std::shuffle(perm.begin(), perm.end(), rng);
            ggml_backend_tensor_set(ids, perm.data(), 0, k * sizeof(int));
            auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            auto t1 = std::chrono::steady_clock::now();
            if (it >= 5) t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(t.begin(), t.end());
        double med = t[t.size() / 2], p90 = t[(size_t) (t.size() * 0.9)], mn = t[0];
        printf("%s k=%2d threads=%d: median %.3f ms  p90 %.3f ms  min %.3f ms  -> %.1f GB/s weight stream (median)\n",
               bname.c_str(), k, threads, med, p90, mn, k * rec / med / 1e6);
        fflush(stdout);
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }
    if (serve_port) {
        int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(serve_port); a.sin_addr.s_addr = INADDR_ANY;
        if (bind(ls, (sockaddr *) &a, sizeof a) || listen(ls, 4)) { perror("bind"); return 1; }
        fprintf(stderr, "serving on %d\n", serve_port);
        auto rd = [](int fd, void * p, size_t n) { char * c = (char *) p; while (n) { ssize_t r = recv(fd, c, n, 0); if (r <= 0) return false; c += r; n -= r; } return true; };
        auto wr = [](int fd, const void * p, size_t n) { const char * c = (const char *) p; while (n) { ssize_t r = send(fd, c, n, 0); if (r <= 0) return false; c += r; n -= r; } return true; };
        std::vector<float> xin(D), rows(16 * D);
        for (;;) {
            int c = accept(ls, nullptr, nullptr);
            setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            int32_t hdr[17];
            while (rd(c, hdr, 17 * 4) && rd(c, xin.data(), D * 4)) {
                int k = hdr[0];
                if (k < 1 || k > 16) break;
                G & gr = graphs[k];
                for (int i = 0; i < k; i++) hdr[1 + i] %= E;
                ggml_backend_tensor_set(gr.ids, hdr + 1, 0, k * sizeof(int));
                ggml_backend_tensor_set(gr.x, xin.data(), 0, D * sizeof(float));
                ggml_backend_graph_compute(be, gr.gf);
                ggml_backend_tensor_get(gr.dn, rows.data(), 0, (size_t) k * D * sizeof(float));
                if (!wr(c, rows.data(), (size_t) k * D * sizeof(float))) break;
            }
            close(c);
        }
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(be);
    return 0;
}
