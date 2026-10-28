// k3ref: scalar CPU reference of Kimi K3's routed-expert FFN for one decode token, bit-exact to the Spark's
// ggml-cuda kernels (sm_121a, -use_fast_math). Implements ref/SPEC.md literally: FTZ on every add/mul/fma, explicit
// fmaf where the SASS has FFMA, MUFU.TANH/EX2/RCP from the exhaustively verified tables (mufu_emu.h).
//
// build: g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math -pthread -march=x86-64-v3 -static -I <llama.cpp>/ggml/src -I ../mufu k3ref.cpp -o k3ref
// usage: k3ref validate MUFU_DIR RECORDS N_RECORDS TG TU TD DUMP_DIR [threads]
//        k3ref bench    MUFU_DIR RECORDS N_RECORDS TG TU TD threads iters
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <atomic>

static int g_threads = 8;
// static-schedule parallel for over [0, n) on g_threads std::threads (no OpenMP: the binary is linked statically)
template <class Fn> static void pfor(int n, bool par, Fn fn) {
    if (!par || g_threads <= 1) { for (int i = 0; i < n; i++) fn(i); return; }
    std::vector<std::thread> th;
    for (int t = 0; t < g_threads; t++) th.emplace_back([=, &fn] { for (int i = (int) ((long) n * t / g_threads); i < (int) ((long) n * (t + 1) / g_threads); i++) fn(i); });
    for (auto & x : th) x.join();
}

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_SYCL   // plain static tables incl. iq1s_grid_gpu (the CUDA layout), no CUDA types
#include "ggml-common.h"   // from <llama.cpp>/ggml/src
#include "mufu_emu.h"

static const int D = 3584, F = 3072, K = 16;

// ---------------------------------------------------------------- float primitives (FTZ, RN)
static inline uint32_t fb(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static inline float bf(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static inline float ftz(float x) { uint32_t u = fb(x); return (u & 0x7f800000u) == 0 ? bf(u & 0x80000000u) : x; }
static inline float fmul(float a, float b) { return ftz(ftz(a) * ftz(b)); }
static inline float fadd(float a, float b) { return ftz(ftz(a) + ftz(b)); }
static inline float ffma(float a, float b, float c) { return ftz(fmaf(ftz(a), ftz(b), ftz(c))); }
// FADD.FTZ.RZ: exact sum in double (both operands are floats of bounded exponent gap here), then toward zero
static inline float fadd_rz(float a, float b) {
    a = ftz(a); b = ftz(b);
    double s = (double) a + (double) b;
    float r = (float) s;
    if (std::isfinite(r) && std::fabs((double) r) > std::fabs(s)) r = std::nextafter(r, 0.0f);
    return ftz(r);
}
static inline float fmaxftz(float a, float b) { a = ftz(a); b = ftz(b); if (std::isnan(a)) return b; if (std::isnan(b)) return a; return a > b ? a : b; }
static inline float h2f(uint16_t h) { _Float16 v; memcpy(&v, &h, 2); return (float) v; }
static inline uint16_t f2h(float f) { _Float16 v = (_Float16) f; uint16_t h; memcpy(&h, &v, 2); return h; }  // RNE
static inline int32_t f2i_trunc(float t) {  // F2I.FTZ.TRUNC.NTZ (S32): NaN -> 0, saturating
    if (std::isnan(t)) return 0;
    if (t >= 2147483648.0f) return INT32_MAX;
    if (t <= -2147483648.0f) return INT32_MIN;
    return (int32_t) t;
}

static mufu_tables MT;
static inline float mufu_rcp(float x)  { return bf(mufu_rcp_bits(&MT, fb(x))); }
static inline float mufu_tanh(float x) { return bf(mufu_tanh_bits(&MT, fb(x))); }
static inline float mufu_ex2(float x)  { return bf(mufu_ex2_bits(&MT, fb(x))); }

// ---------------------------------------------------------------- quantize_q8_1 (SPEC 2)
struct Q8 { int8_t qs[32]; uint16_t d, s; };

static void quantize_q8_1(const float * x, int n, Q8 * y) {
    for (int b = 0; b < n / 32; b++) {
        const float * xb = x + 32 * b;
        float am[32], sm[32];
        for (int i = 0; i < 32; i++) { am[i] = fadd(std::fabs(xb[i]), -0.0f); sm[i] = xb[i]; }
        for (int off = 16; off > 0; off >>= 1) {
            float na[32], ns[32];
            for (int i = 0; i < 32; i++) { na[i] = fmaxftz(am[i], am[i ^ off]); ns[i] = fadd(sm[i], sm[i ^ off]); }
            memcpy(am, na, sizeof am); memcpy(sm, ns, sizeof sm);
        }
        const float amax = am[0], sum = sm[0];
        const float d = fmul(amax, bf(0x3C010204u));
        const bool nz = !(amax == 0.0f);   // FSETP.NEU (unordered -> true)
        for (int i = 0; i < 32; i++) {
            int32_t q = 0;
            if (nz) {
                const float r = mufu_rcp(d);
                const float v = fmul(r, xb[i]);
                const float c = bf(0x3f000000u | (fb(v) & 0x80000000u));
                const float t = std::trunc(fadd_rz(v, c));
                q = f2i_trunc(t);
            }
            y[b].qs[i] = (int8_t) (uint8_t) (q & 0xff);
        }
        y[b].d = f2h(d);
        y[b].s = f2h(sum);
    }
}

// ---------------------------------------------------------------- vec_dot (SPEC 3), one thread's partial
static inline int dp4a(uint32_t a, uint32_t b, int c) {
    for (int i = 0; i < 4; i++) c += (int) (int8_t) (a >> (8 * i)) * (int) (int8_t) (b >> (8 * i));
    return c;
}
static inline uint32_t ld32(const void * p) { uint32_t v; memcpy(&v, p, 4); return v; }

static float vd_iq1_s(const block_iq1_s * blk, const Q8 & y, int j) {
    const uint32_t qsp = ld32(blk->qs + 4 * j);
    const uint8_t * qs = (const uint8_t *) &qsp;
    const int qh = blk->qh[j];
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t grid = iq1s_grid_gpu[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const uint32_t g0 = grid & 0x0F0F0F0Fu, g1 = (grid >> 4) & 0x0F0F0F0Fu;
        sumi = dp4a(g0, ld32(y.qs + 4 * l0), sumi);
        sumi = dp4a(g1, ld32(y.qs + 4 * (l0 + 1)), sumi);
    }
    const float d1q = fmul(h2f(blk->d), (float) (((qh >> 11) & 0x0E) + 1));
    const float delta = (qh & 0x8000) ? -1.125f : -0.875f;
    const float inner = ffma(h2f(y.d), (float) sumi, fmul(h2f(y.s), delta));
    return ffma(d1q, inner, 0.0f);
}

static float vd_iq2_xxs(const block_iq2_xxs * blk, const Q8 & y, int j) {
    const int iqs = 2 * j;
    // get_int_b2(qs, iqs) reads bytes 4*iqs .. 4*iqs+3
    const uint8_t * qb = (const uint8_t *) blk->qs;
    const uint32_t q2v = ld32(qb + 4 * iqs);
    const uint32_t aux32 = ld32(qb + 4 * (iqs + 1));
    const uint8_t * aux8 = (const uint8_t *) &q2v;
    int sumi = 0;
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint64_t gp = iq2xxs_grid[aux8[k0 / 2]];
        const uint32_t gx = (uint32_t) gp, gy = (uint32_t) (gp >> 32);
        const uint8_t v = (uint8_t) (aux32 >> (7 * k0 / 2));
        const uint32_t p = __builtin_popcount(v) & 1;
        const uint32_t s = (uint32_t) (v ^ (p << 7)) * 0x01010101u;
        uint32_t g0 = 0, g1 = 0;
        for (int b = 0; b < 4; b++) {
            const uint32_t m0 = (s & (0x08040201u & (0xffu << 8 * b))) ? 0xffu : 0u;
            const uint32_t m1 = (s & (0x80402010u & (0xffu << 8 * b))) ? 0xffu : 0u;
            const uint32_t x0 = (gx >> 8 * b) & 0xff, x1 = (gy >> 8 * b) & 0xff;
            g0 |= (((x0 ^ m0) - m0) & 0xffu) << 8 * b;
            g1 |= (((x1 ^ m1) - m1) & 0xffu) << 8 * b;
        }
        sumi = dp4a(g0, ld32(y.qs + 4 * k0), sumi);
        sumi = dp4a(g1, ld32(y.qs + 4 * (k0 + 1)), sumi);
    }
    const int ls = (int) (aux32 >> 27 | 1);
    sumi = sumi * ls / 8;
    const float d = fmul(h2f(blk->d), h2f(y.d));
    return ffma(d, (float) sumi, 0.0f);
}

// one output row: 128 virtual threads, then ((w0+w1)+w2)+w3 per lane, then the xor butterfly
static float mmvq_row(int type, const uint8_t * row, int nb, const Q8 * y) {
    float p[128];
    for (int t = 0; t < 128; t++) {
        const int kb = t / 8, j = t % 8;
        if (kb >= nb) { p[t] = 0.0f; continue; }
        if (type == 19) p[t] = vd_iq1_s((const block_iq1_s *) row + kb, y[kb * 8 + j], j);
        else            p[t] = vd_iq2_xxs((const block_iq2_xxs *) row + kb, y[kb * 8 + j], j);
    }
    float s[32];
    for (int l = 0; l < 32; l++) s[l] = fadd(fadd(fadd(p[32 + l], p[l]), p[64 + l]), p[96 + l]);
    for (int off = 16; off > 0; off >>= 1) {
        float n[32];
        for (int l = 0; l < 32; l++) n[l] = fadd(s[l], s[l ^ off]);
        memcpy(s, n, sizeof s);
    }
    return s[0];
}

static size_t row_bytes(int type, int k) { return (size_t) (k / 256) * (type == 19 ? sizeof(block_iq1_s) : sizeof(block_iq2_xxs)); }

struct Layer { int tg, tu, td; size_t gsz, usz, dsz, rec; const uint8_t * base; };

// one expert: x (already quantized as qx) -> g, u, h, dn
static void expert(const Layer & L, int slot, const Q8 * qx, float * g, float * u, float * h, float * dn, bool par) {
    const uint8_t * r = L.base + (size_t) slot * L.rec;
    const size_t rbg = row_bytes(L.tg, D), rbu = row_bytes(L.tu, D), rbd = row_bytes(L.td, F);
    pfor(F, par, [&](int i) {
        g[i] = mmvq_row(L.tg, r + i * rbg, D / 256, qx);
        u[i] = mmvq_row(L.tu, r + L.gsz + i * rbu, D / 256, qx);
        const float a = fmul(mufu_tanh(fmul(g[i], 0.25f)), 4.0f);
        const float e = mufu_ex2(fmul(g[i], bf(0xBFB8AA3Bu)));
        const float s = mufu_rcp(fadd(e, 1.0f));
        const float b = fmul(s, a);
        const float c = fmul(mufu_tanh(fmul(u[i], bf(0x3D23D70Au))), 25.0f);
        h[i] = fmul(b, c);
    });
    std::vector<Q8> qh(F / 32);
    quantize_q8_1(h, F, qh.data());
    pfor(D, par, [&](int i) { dn[i] = mmvq_row(L.td, r + L.gsz + L.usz + i * rbd, F / 256, qh.data()); });
}

static const void * map(const std::string & p, size_t * n = nullptr) {
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) { perror(p.c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    void * m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    if (n) *n = st.st_size;
    return m;
}

int main(int argc, char ** argv) {
    if (argc < 9) { fprintf(stderr, "usage: see source\n"); return 1; }
    std::string mode = argv[1], md = argv[2];
    MT.rcp = (const uint32_t *) map(md + "/rcp.bin");
    MT.tanh_ = (const uint32_t *) map(md + "/tanh.bin");
    MT.ex2p = (const uint32_t *) map(md + "/ex2p.bin");
    MT.ex2n = (const uint32_t *) map(md + "/ex2n.bin");
    Layer L;
    const int E = atoi(argv[4]);
    L.tg = atoi(argv[5]); L.tu = atoi(argv[6]); L.td = atoi(argv[7]);
    L.gsz = row_bytes(L.tg, D) * F; L.usz = row_bytes(L.tu, D) * F; L.dsz = row_bytes(L.td, F) * D;
    size_t fsz; L.base = (const uint8_t *) map(argv[3], &fsz);
    L.rec = fsz / E;

    if (mode == "validate") {
        std::string dd = argv[8];
        if (argc > 9) g_threads = atoi(argv[9]);
        size_t nx; const float * X = (const float *) map(dd + "/x.bin", &nx);
        const int ncase = (int) (nx / (4 * D));
        const int32_t * IDS = (const int32_t *) map(dd + "/ids.bin");
        const float * G = (const float *) map(dd + "/g.bin"), * U = (const float *) map(dd + "/u.bin");
        const float * H = (const float *) map(dd + "/h.bin"), * DN = (const float *) map(dd + "/dn.bin");
        unsigned long long bad[4] = {0, 0, 0, 0}, tot[4] = {0, 0, 0, 0}, badrows = 0, rows = 0;
        int firstbad[4] = {-1, -1, -1, -1};
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> g(F * K), u(F * K), h(F * K), dn(D * K);
        for (int c = 0; c < ncase; c++) {
            std::vector<Q8> qx(D / 32);
            quantize_q8_1(X + (size_t) c * D, D, qx.data());
            pfor(K, true, [&](int k) { expert(L, IDS[c * K + k], qx.data(), &g[k * F], &u[k * F], &h[k * F], &dn[k * D], false); });
            const float * ref[4] = {G + (size_t) c * K * F, U + (size_t) c * K * F, H + (size_t) c * K * F, DN + (size_t) c * K * D};
            const float * mine[4] = {g.data(), u.data(), h.data(), dn.data()};
            const int n[4] = {F * K, F * K, F * K, D * K};
            for (int s = 0; s < 4; s++) {
                for (int i = 0; i < n[s]; i++) if (fb(mine[s][i]) != fb(ref[s][i])) { bad[s]++; if (firstbad[s] < 0) firstbad[s] = c; }
                tot[s] += n[s];
            }
            for (int k = 0; k < K; k++) { rows++; if (memcmp(&dn[k * D], ref[3] + k * D, 4 * D)) badrows++; }
        }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const char * nm[4] = {"g", "u", "h", "dn"};
        for (int s = 0; s < 4; s++) printf("%-2s: %llu / %llu values bit-identical (%llu differ, first case %d)\n", nm[s], tot[s] - bad[s], tot[s], bad[s], firstbad[s]);
        printf("dn rows: %llu / %llu bit-identical; %d cases in %.1f s\n", rows - badrows, rows, ncase, secs);
        return bad[3] ? 2 : 0;
    }
    if (mode == "bench") {
        const int threads = atoi(argv[8]), iters = argc > 9 ? atoi(argv[9]) : 50;
        g_threads = threads;
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x(D); for (auto & v : x) v = nd(rng);
        std::vector<float> g(F), u(F), h(F), dn(D);
        std::vector<int> perm(E);
        for (int k = 1; k <= 4; k++) {
            std::vector<double> t;
            for (int it = 0; it < iters + 3; it++) {
                for (int i = 0; i < E; i++) perm[i] = i;
                std::shuffle(perm.begin(), perm.end(), rng);
                auto t0 = std::chrono::steady_clock::now();
                std::vector<Q8> qx(D / 32);
                quantize_q8_1(x.data(), D, qx.data());
                for (int j = 0; j < k; j++) expert(L, perm[j], qx.data(), g.data(), u.data(), h.data(), dn.data(), true);
                if (it >= 3) t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(t.begin(), t.end());
            printf("k=%d threads=%d: median %.2f ms p90 %.2f ms min %.2f ms\n", k, threads, t[t.size() / 2], t[t.size() * 9 / 10], t[0]);
        }
        return 0;
    }
    return 1;
}
