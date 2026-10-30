// glmref: fp64 reference for the 8 experts of one layer slice (weights dequantized with ggml's to_float), compared
// against glmbench --check dumps. usage: glmref PREFIX TG TU TD DUMP1 [DUMP2 ...]   (DUMPn = path without .x/.dn)
// Per dump: max |dn - ref| over all 8 x 6144 rows, that over max |ref| (relative), worst row cosine, and the same
// against DUMP1 (backend vs backend).
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int D = 6144, F = 2048, NE = 8;

static std::vector<char> slurp(const std::string & p, size_t n) {
    std::vector<char> b(n);
    FILE * f = fopen(p.c_str(), "rb");
    if (!f || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "read %s failed\n", p.c_str()); exit(1); }
    fclose(f);
    return b;
}

static std::vector<float> deq(ggml_type t, const std::vector<char> & b, int rows, int cols) {
    std::vector<float> o((size_t) rows * cols);
    size_t rs = ggml_row_size(t, cols);
    auto * tr = ggml_get_type_traits(t);
    for (int r = 0; r < rows; r++) tr->to_float(b.data() + r * rs, o.data() + (size_t) r * cols, cols);
    return o;
}

struct Cmp { double maxabs = 0, refmax = 0, mincos = 1, va = 0, vr = 0; int at = 0; };
static Cmp cmp(const float * a, const double * r) {
    Cmp c;
    for (int e = 0; e < NE; e++) {
        double dot = 0, na = 0, nr = 0;
        for (int i = 0; i < D; i++) {
            double x = a[e * D + i], y = r[e * D + i];
            if (std::fabs(x - y) > c.maxabs) { c.maxabs = std::fabs(x - y); c.at = e * D + i; c.va = x; c.vr = y; }
            c.refmax = std::max(c.refmax, std::fabs(y));
            dot += x * y; na += x * x; nr += y * y;
        }
        c.mincos = std::min(c.mincos, dot / std::sqrt(na * nr));
    }
    return c;
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: see source\n"); return 1; }
    ggml_cpu_init();
    std::string pre = argv[1];
    ggml_type tg = (ggml_type) atoi(argv[2]), tu = (ggml_type) atoi(argv[3]), td = (ggml_type) atoi(argv[4]);
    size_t gsz = ggml_row_size(tg, D) * F, usz = ggml_row_size(tu, D) * F, dsz = ggml_row_size(td, F) * D;
    auto wg = deq(tg, slurp(pre + "_gate.bin", gsz * NE), F * NE, D);
    auto wu = deq(tu, slurp(pre + "_up.bin", usz * NE), F * NE, D);
    auto wd = deq(td, slurp(pre + "_down.bin", dsz * NE), D * NE, F);
    std::string d1 = argv[5];
    auto xb = slurp(d1 + ".x", D * 4);
    const float * x = (const float *) xb.data();
    std::vector<double> ref((size_t) NE * D), h(F), gref((size_t) NE * F), href((size_t) NE * F);
    for (int e = 0; e < NE; e++) {
        for (int j = 0; j < F; j++) {
            double g = 0, u = 0;
            const float * rg = &wg[((size_t) e * F + j) * D], * ru = &wu[((size_t) e * F + j) * D];
            for (int i = 0; i < D; i++) { g += (double) rg[i] * x[i]; u += (double) ru[i] * x[i]; }
            h[j] = g / (1 + std::exp(-g)) * u; gref[(size_t) e * F + j] = g; href[(size_t) e * F + j] = h[j];
        }
        for (int i = 0; i < D; i++) {
            double s = 0;
            const float * rd = &wd[((size_t) e * D + i) * F];
            for (int j = 0; j < F; j++) s += (double) rd[j] * h[j];
            ref[(size_t) e * D + i] = s;
            if (getenv("GLMREF_ROW") && atoi(getenv("GLMREF_ROW")) == e * D + i) {
                double mx = 0; int at = 0;
                for (int j = 0; j < F; j++) if (std::fabs(rd[j]) > mx) { mx = std::fabs(rd[j]); at = j; }
                fprintf(stderr, "row %d: max|w| %.4g at %d, h there %.4g, h absmax", e * D + i, mx, at, h[at]);
                double hm = 0; for (int j = 0; j < F; j++) hm = std::max(hm, std::fabs(h[j])); fprintf(stderr, " %.4g\n", hm);
                {   // the CPU backend's vec_dot on this row with the fp64 h quantized by its from_float
                    auto * ct = ggml_get_type_traits_cpu(td); ggml_type vt = ct->vec_dot_type;
                    std::vector<float> hf(h.begin(), h.end()); std::vector<char> hq(ggml_row_size(vt, F));
                    ggml_get_type_traits_cpu(vt)->from_float(hf.data(), hq.data(), F);
                    auto wb = slurp(pre + "_down.bin", dsz * NE); float r = 0;
                    ct->vec_dot(F, &r, 0, wb.data() + dsz * e + ggml_row_size(td, F) * i, 0, hq.data(), 0, 1);
                    std::vector<float> hd(F);   // q8_K block: f32 d, int8 qs[256], int16 bsums[16]
                    for (int b = 0; b < F / 256; b++) { const char * blk = hq.data() + b * 292; float d; memcpy(&d, blk, 4);
                        for (int j = 0; j < 256; j++) hd[b * 256 + j] = d * (int8_t) blk[4 + j]; }
                    double s2 = 0; for (int j = 0; j < F; j++) s2 += rd[j] * hd[j];
                    fprintf(stderr, "  cpu vec_dot %.5g, fp64 dot with dequantized q8 h %.5g\n", r, s2);
                }
                for (int b = 0; b < F / 256; b++) { double sb = 0, wm = 0; for (int j = b * 256; j < b * 256 + 256; j++) { sb += rd[j] * h[j]; wm = std::max(wm, (double) std::fabs(rd[j])); } fprintf(stderr, "  blk %d sum %.4g max|w| %.4g\n", b, sb, wm); }
            }
        }
    }
    auto exists = [](const std::string & p) { FILE * f = fopen(p.c_str(), "rb"); if (f) fclose(f); return f != nullptr; };
    if (exists(d1 + ".g")) {   // gate stage of DUMP1 vs fp64
        auto b = slurp(d1 + ".g", (size_t) NE * F * 4); const float * gg = (const float *) b.data();
        double ma = 0, mr = 0; int at = 0;
        for (int i = 0; i < NE * F; i++) { if (std::fabs(gg[i] - gref[i]) > ma) { ma = std::fabs(gg[i] - gref[i]); at = i; } mr = std::max(mr, std::fabs(gref[i])); }
        printf("gate stage %s: maxabs %.4g (at %d: %.5g vs %.5g) refmax %.4g\n", argv[5], ma, at, gg[at], gref[at], mr);
    }
    if (exists(d1 + ".h")) {
        auto b = slurp(d1 + ".h", (size_t) NE * F * 4); const float * gg = (const float *) b.data();
        double ma = 0, mr = 0; int at = 0;
        for (int i = 0; i < NE * F; i++) { if (std::fabs(gg[i] - href[i]) > ma) { ma = std::fabs(gg[i] - href[i]); at = i; } mr = std::max(mr, std::fabs(href[i])); }
        printf("h stage %s: maxabs %.4g (at %d: %.5g vs %.5g) refmax %.4g\n", argv[5], ma, at, gg[at], href[at], mr);
    }
    { std::vector<float> rf(ref.begin(), ref.end()); FILE * f = fopen((d1 + ".ref").c_str(), "wb"); fwrite(rf.data(), 4, rf.size(), f); fclose(f); }
    for (int a = 5; a < argc; a++) {   // kernel arithmetic: g/u/dn against fp64 sums over the kernel's own q8_1 inputs
        std::string p = argv[a];
        if (!exists(p + ".xq")) continue;
        auto xq = slurp(p + ".xq", D), xdb = slurp(p + ".xd", D / 32 * 4), hq = slurp(p + ".hq", NE * F), hdb = slurp(p + ".hd", NE * F / 32 * 4);
        auto gub = slurp(p + ".gu", 2 * NE * F * 4), dnb = slurp(p + ".dn", NE * D * 4);
        const float * xd = (const float *) xdb.data(), * hd = (const float *) hdb.data(), * gu = (const float *) gub.data(), * dn = (const float *) dnb.data();
        double eg = 0, eu = 0, ed = 0;   // max |kernel - exact| / max |exact| over each expert's rows
        for (int e = 0; e < NE; e++) {
            double mg = 0, mu = 0, mgv = 0, muv = 0, md = 0, mdv = 0;
            for (int j = 0; j < F; j++) {
                double g = 0, u = 0;
                for (int i = 0; i < D; i++) { double xv = (double) xd[i / 32] * (int8_t) xq[i]; g += wg[((size_t) e * F + j) * D + i] * xv; u += wu[((size_t) e * F + j) * D + i] * xv; }
                mg = std::max(mg, std::fabs(gu[2 * e * F + j] - g)); mgv = std::max(mgv, std::fabs(g));
                mu = std::max(mu, std::fabs(gu[(2 * e + 1) * F + j] - u)); muv = std::max(muv, std::fabs(u));
            }
            for (int i = 0; i < D; i++) {
                double s = 0;
                for (int j = 0; j < F; j++) s += wd[((size_t) e * D + i) * F + j] * (double) hd[(e * F + j) / 32] * (int8_t) hq[e * F + j];
                md = std::max(md, std::fabs(dn[e * D + i] - s)); mdv = std::max(mdv, std::fabs(s));
            }
            eg = std::max(eg, mg / mgv); eu = std::max(eu, mu / muv); ed = std::max(ed, md / mdv);
        }
        printf("%-48s kernel vs exact sum over its q8_1 inputs: gate %.2g up %.2g down %.2g (max err / row-set max)\n", argv[a], eg, eu, ed);
    }
    std::vector<std::vector<float>> dumps;
    for (int a = 5; a < argc; a++) {
        auto b = slurp(std::string(argv[a]) + ".dn", (size_t) NE * D * 4);
        std::vector<float> v((const float *) b.data(), (const float *) b.data() + NE * D);
        Cmp c = cmp(v.data(), ref.data());
        std::vector<double> d0(dumps.empty() ? v.begin() : dumps[0].begin(), dumps.empty() ? v.end() : dumps[0].end());
        Cmp c1 = cmp(v.data(), d0.data());
        printf("%-48s vs fp64: maxabs %.4g rel %.3g mincos %.7f (worst [%d] %.5g vs %.5g) | vs %s: maxabs %.4g rel %.3g\n", argv[a], c.maxabs,
               c.maxabs / c.refmax, c.mincos, c.at, c.va, c.vr, argv[5], c1.maxabs, c1.maxabs / c1.refmax);
        dumps.push_back(v);
    }
    return 0;
}
