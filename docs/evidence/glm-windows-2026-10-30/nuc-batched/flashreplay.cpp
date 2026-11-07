// Replay remote rows dumped by the engine (LLAMA_EXPERT_STORE_REMOTE_DUMP) through the production ggml CUDA kernels:
// for every (gid, input, row) the GPU computes the expert's row from the store record and compares it bit for bit.
// usage: flashreplay <store.k3xs> <bases.txt: "layer base_offset" lines> <dump.bin>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s store bases dump\n", argv[0]); return 1; }
    const int D = 4096, F = 2048;
    const size_t gsz = 2424832, usz = 2424832, dsz = 3211264, R = gsz + usz + dsz;
    std::map<int, long long> base;
    { FILE * f = fopen(argv[2], "r"); int l; long long b; while (fscanf(f, "%d %lld", &l, &b) == 2) base[l] = b; fclose(f); }
    FILE * df = fopen(argv[3], "rb");
    int hdr[3];
    if (!df || fread(hdr, 4, 3, df) != 3 || hdr[1] != D || hdr[2] != D) { fprintf(stderr, "bad dump\n"); return 1; }
    const int n = hdr[0];
    const size_t rec = 2 + (size_t) D + D;
    std::vector<float> all((size_t) n * rec);
    if (fread(all.data(), 4, all.size(), df) != all.size()) { fprintf(stderr, "short dump\n"); return 1; }
    fclose(df);
    FILE * st = fopen(argv[1], "rb");

    ggml_backend_t be = ggml_backend_cuda_init(0);
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * wg = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XS, D, F, 1);
    ggml_tensor * wu = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XS, D, F, 1);
    ggml_tensor * wd = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ3_XXS, F, D, 1);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    ggml_tensor * g = ggml_mul_mat_id(ctx, wg, x, ids);
    ggml_tensor * u = ggml_mul_mat_id(ctx, wu, x, ids);
    ggml_tensor * h = ggml_swiglu_clamp(ctx, g, u, 10.0f);
    ggml_tensor * o = ggml_mul_mat_id(ctx, wd, h, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    const int32_t zero = 0;
    ggml_backend_tensor_set(ids, &zero, 0, 4);
    std::vector<uint8_t> r(R);
    std::vector<float> out(D);
    int bad = 0, last_gid = -1;
    for (int i = 0; i < n; i++) {
        const float * p = all.data() + (size_t) i * rec;
        const int gid = (int) p[0], L = gid / 288, e = gid % 288;
        if (gid != last_gid) {
            if (!base.count(L)) { fprintf(stderr, "no base for layer %d\n", L); return 1; }
            if (fseeko(st, base[L] + (long long) e * R, SEEK_SET) || fread(r.data(), 1, R, st) != R) { fprintf(stderr, "read\n"); return 1; }
            ggml_backend_tensor_set(wg, r.data(), 0, gsz);
            ggml_backend_tensor_set(wu, r.data() + gsz, 0, usz);
            ggml_backend_tensor_set(wd, r.data() + gsz + usz, 0, dsz);
            last_gid = gid;
        }
        ggml_backend_tensor_set(x, p + 2, 0, D * 4);
        ggml_backend_graph_compute(be, gf);
        ggml_backend_tensor_get(o, out.data(), 0, D * 4);
        int nb = 0, first = -1;
        for (int j = 0; j < D; j++) if (memcmp(&out[j], p + 2 + D + j, 4)) { nb++; if (first < 0) first = j; }
        if (nb) {
            if (bad < 10) fprintf(stderr, "row %d gid %d (layer %d expert %d) step %d: %d values differ, first [%d] gpu %.9g remote %.9g\n",
                                  i, gid, L, e, (int) p[1], nb, first, out[first], p[2 + D + first]);
            bad++;
        }
    }
    printf("replay %s: %d remote rows, %d differ from the GPU's rows\n", argv[3], n, bad);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return bad ? 2 : 0;
}
