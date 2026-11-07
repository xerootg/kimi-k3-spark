// GPU reference rows for the GLM-5.3-Flash routed experts, computed by the production ggml CUDA kernels exactly the way
// the model builds them (mul_mat_id gate, mul_mat_id up, swiglu_clamp 10, mul_mat_id down; the CUDA backend fuses
// gate+up+GLU when adjacent). For checking a NUC implementation bit for bit.
// usage: flashrow <store.k3xs> <out.bin> <T tokens> <ncases> <rec_off x8>   (records: IQ2_XS gate|up, IQ3_XXS down)
// out.bin: int32 ncases, T, 8; per case: x[T][4096] f32, ids[T][8] i32 (index into the 8 records), out[T][8][4096] f32
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 13) { fprintf(stderr, "usage: %s store out T ncases off*8\n", argv[0]); return 1; }
    const int D = 4096, F = 2048, NE = 8;
    const int T = atoi(argv[3]), ncases = atoi(argv[4]);
    const size_t gsz = 2424832, usz = 2424832, dsz = 3211264;
    FILE * st = fopen(argv[1], "rb");
    if (!st) { perror("store"); return 1; }
    std::vector<uint8_t> G(NE * gsz), U(NE * usz), Dn(NE * dsz);
    for (int e = 0; e < NE; e++) {
        const long long off = atoll(argv[5 + e]);
        if (fseeko(st, off, SEEK_SET) || fread(&G[e * gsz], 1, gsz, st) != gsz || fread(&U[e * usz], 1, usz, st) != usz ||
            fread(&Dn[e * dsz], 1, dsz, st) != dsz) { fprintf(stderr, "read rec %d\n", e); return 1; }
    }
    fclose(st);

    ggml_backend_t be = ggml_backend_cuda_init(0);
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * wg = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XS, D, F, NE);
    ggml_tensor * wu = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XS, D, F, NE);
    ggml_tensor * wd = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ3_XXS, F, D, NE);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NE, T);
    ggml_tensor * g = ggml_mul_mat_id(ctx, wg, x, ids);
    ggml_tensor * u = ggml_mul_mat_id(ctx, wu, x, ids);
    ggml_tensor * h = ggml_swiglu_clamp(ctx, g, u, 10.0f);
    ggml_tensor * o = ggml_mul_mat_id(ctx, wd, h, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(wg, G.data(), 0, G.size());
    ggml_backend_tensor_set(wu, U.data(), 0, U.size());
    ggml_backend_tensor_set(wd, Dn.data(), 0, Dn.size());

    FILE * out = fopen(argv[2], "wb");
    int hdr[3] = { ncases, T, NE };
    fwrite(hdr, 4, 3, out);
    std::mt19937 rng(4242);
    std::vector<float> xv((size_t) D * T), ov((size_t) D * NE * T);
    std::vector<int32_t> iv((size_t) NE * T);
    const float scales[] = { 0.02f, 0.1f, 0.3f, 1.0f, 2.0f, 5.0f, 20.0f };
    for (int c = 0; c < ncases; c++) {
        const float s = scales[c % 7];
        std::normal_distribution<float> nd(0.f, s);
        for (auto & v : xv) v = nd(rng);
        if (c % 11 == 5) for (int i = 0; i < D; i += 3) xv[i] = 0.0f;                  // zeros
        if (c % 13 == 7) xv[rng() % D] = 300.0f;                                        // an outlier
        for (int t = 0; t < T; t++) {
            std::vector<int> p = { 0, 1, 2, 3, 4, 5, 6, 7 };
            std::shuffle(p.begin(), p.end(), rng);
            for (int k = 0; k < NE; k++) iv[t * NE + k] = p[k];
        }
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
        ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * 4);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        ggml_backend_tensor_get(o, ov.data(), 0, ov.size() * 4);
        fwrite(xv.data(), 4, xv.size(), out);
        fwrite(iv.data(), 4, iv.size(), out);
        fwrite(ov.data(), 4, ov.size(), out);
    }
    fclose(out);
    fprintf(stderr, "wrote %d cases, T=%d\n", ncases, T);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
