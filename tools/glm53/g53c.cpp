// g53c: one GLM-5.3 token's routed-expert FFN (k experts) on the CPU with AVX2 + AVX-VNNI (vpdpbusd), standalone.
// Same activation quantization as g53x (q8_1: 32-blocks, d = amax/127, round half away, xs = d * sum q), ggml's
// weight blocks unchanged. Gate/up/down for IQ2_XS, IQ3_XXS, IQ4_XS, Q4_K, Q5_K (Q6_K not yet).
//
//   g53c PREFIX TG TU TD THREADS bench k1,k2,... ITERS
//   g53c PREFIX TG TU TD THREADS check OUT          ids 0..7; OUT.{x,gu,xq,xd,hq,hd,dn} for glmref
//
// Work: phase 1 = (expert, 32 gate/up rows) tasks; a task also writes the q8_1 block of h it produced, so phase 2
// (expert, 64 down rows) starts after one barrier. Workers spin on a generation counter (no sleeping between tokens).
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

static const int D = 6144, F = 2048, NE = 8, SLOTS = 64;
enum { T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ4_XS = 23 };
static size_t bbytes(int t) {
    switch (t) { case T_Q4_K: return 144; case T_Q5_K: return 176; case T_IQ2_XS: return 74; case T_IQ3_XXS: return 98; case T_IQ4_XS: return 136; }
    fprintf(stderr, "type %d not supported on the CPU\n", t); exit(1);
}
static inline float h2f(const uint8_t * p) { uint16_t u; memcpy(&u, p, 2); return _cvtsh_ss(u); }
static inline float hsum(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a)); a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}
#ifndef G53C_GATHER
#define G53C_GATHER 0   // 1 = vpgather for the IQ2_XS grid / sign masks: slower on this laptop (0.6 -> 0.8 ms/expert)
#endif
#ifndef G53C_G3GATHER
#define G53C_G3GATHER 0   // 1 = vpgatherdd for the IQ3_XXS grid: ~8 % slower here
#endif
#define DP(a, b) _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), a, b)

struct Act { const int8_t * q; const float * d, * s; const int * si; };   // q8_1 vector: values, scale, d*sum, int sum

// row . x for one row of n values (n / 256 super-blocks)
template <int T> static float row_dot(const uint8_t * row, int n, const Act & x);

template <> float row_dot<T_IQ2_XS>(const uint8_t * row, int n, const Act & x) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i one = _mm256_set1_epi8(1);
    for (int sb = 0; sb < n / 256; sb++) {
        const uint8_t * b = row + sb * 74; const float d = h2f(b) * 0.25f;
        const uint16_t * qs = (const uint16_t *) (b + 2);
        for (int ib = 0; ib < 8; ib++) {
            const int u = sb * 8 + ib;
#if G53C_GATHER   // 4 x 64-bit gathers for the grid and the sign masks (no GDS penalty on Meteor/Arrow Lake)
            const __m256i qi = _mm256_cvtepu16_epi64(_mm_loadl_epi64((const __m128i *) (qs + 4 * ib)));
            const __m256i g = _mm256_i64gather_epi64((const long long *) iq2xs_grid, _mm256_and_si256(qi, _mm256_set1_epi64x(511)), 8);
            const __m256i s = _mm256_or_si256(_mm256_i64gather_epi64((const long long *) ksigns64, _mm256_srli_epi64(qi, 9), 8), one);
#else
            const uint16_t q0 = qs[4 * ib], q1 = qs[4 * ib + 1], q2 = qs[4 * ib + 2], q3 = qs[4 * ib + 3];
            const __m256i g = _mm256_set_epi64x(iq2xs_grid[q3 & 511], iq2xs_grid[q2 & 511], iq2xs_grid[q1 & 511], iq2xs_grid[q0 & 511]);
            const __m256i s = _mm256_or_si256(_mm256_set_epi64x(ksigns64[q3 >> 9], ksigns64[q2 >> 9], ksigns64[q1 >> 9], ksigns64[q0 >> 9]), one);
#endif
            const __m256i y = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (x.q + u * 32)), s);
            const uint8_t sc = b[66 + ib]; const float f = d * x.d[u];
            const __m256 sv = _mm256_setr_m128(_mm_set1_ps(f * (0.5f + (sc & 15))), _mm_set1_ps(f * (0.5f + (sc >> 4))));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(DP(g, y)), sv, acc);
        }
    }
    return hsum(acc);
}
template <> float row_dot<T_IQ3_XXS>(const uint8_t * row, int n, const Act & x) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i one = _mm256_set1_epi8(1);
    for (int sb = 0; sb < n / 256; sb++) {
        const uint8_t * b = row + sb * 98; const float d = h2f(b) * 0.5f;
        for (int ib = 0; ib < 8; ib++) {
            const int u = sb * 8 + ib;
            const uint8_t * q3 = b + 2 + 8 * ib; uint32_t aux; memcpy(&aux, b + 66 + 4 * ib, 4);
            const __m256i idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) q3));
#if G53C_G3GATHER
            const __m256i g = _mm256_i32gather_epi32((const int *) iq3xxs_grid, idx, 4);
#else
            const __m256i g = _mm256_setr_epi32(iq3xxs_grid[q3[0]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[3]],
                                                iq3xxs_grid[q3[4]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[7]]);
            (void) idx;
#endif
#if G53C_GATHER
            const __m128i si = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) aux), _mm_setr_epi32(0, 7, 14, 21)), _mm_set1_epi32(127));
            const __m256i s = _mm256_or_si256(_mm256_i32gather_epi64((const long long *) ksigns64, si, 8), one);
#else
            const __m256i s = _mm256_or_si256(_mm256_set_epi64x(ksigns64[(aux >> 21) & 127], ksigns64[(aux >> 14) & 127],
                                                                ksigns64[(aux >> 7) & 127], ksigns64[aux & 127]), one);
#endif
            const __m256i y = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (x.q + u * 32)), s);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(DP(g, y)), _mm256_set1_ps(d * (0.5f + (aux >> 28)) * x.d[u]), acc);
        }
    }
    return hsum(acc);
}
template <> float row_dot<T_IQ4_XS>(const uint8_t * row, int n, const Act & x) {
    __m256 acc = _mm256_setzero_ps(); float corr = 0.f;
    __m128i t = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    t = _mm_add_epi8(t, _mm_set1_epi8((char) 128));   // unsigned kv + 128; subtract 128 * sum(y) per block
    const __m256i tbl = _mm256_set_m128i(t, t), m4 = _mm256_set1_epi8(15);
    for (int sb = 0; sb < n / 256; sb++) {
        const uint8_t * b = row + sb * 136; const float d = h2f(b); uint16_t sh; memcpy(&sh, b + 2, 2);
        for (int ib = 0; ib < 8; ib++) {
            const int u = sb * 8 + ib;
            const int ls = ((b[4 + ib / 2] >> 4 * (ib & 1)) & 15) | (((sh >> 2 * ib) & 3) << 4);
            const __m128i q = _mm_loadu_si128((const __m128i *) (b + 8 + 16 * ib));
            const __m256i nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), m4);
            const __m256i v = _mm256_shuffle_epi8(tbl, nib);
            const float f = d * (ls - 32) * x.d[u];
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(DP(v, _mm256_loadu_si256((const __m256i *) (x.q + u * 32)))), _mm256_set1_ps(f), acc);
            corr += f * 128.f * x.si[u];
        }
    }
    return hsum(acc) - corr;
}
static inline void k4sm(const uint8_t * sc, int j, int & s, int & m) {
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
    else { s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
}
template <int T5> static float row_dot_k(const uint8_t * row, int n, const Act & x) {
    __m256 acc = _mm256_setzero_ps(); float macc = 0.f;
    const __m256i m4 = _mm256_set1_epi8(15), m1 = _mm256_set1_epi8(1);
    const int bsz = T5 ? 176 : 144;
    for (int sb = 0; sb < n / 256; sb++) {
        const uint8_t * b = row + sb * bsz; const float d = h2f(b), dmin = h2f(b + 2);
        const __m256i qh = T5 ? _mm256_loadu_si256((const __m256i *) (b + 16)) : _mm256_setzero_si256();
        for (int p = 0; p < 4; p++) {
            const int blk = sb * 8 + 2 * p;
            const __m256i q = _mm256_loadu_si256((const __m256i *) (b + (T5 ? 48 : 16) + 32 * p));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            if (T5) {
                lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * p), m1), 4));
                hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * p + 1), m1), 4));
            }
            int s0, mn0, s1, mn1; k4sm(b + 4, 2 * p, s0, mn0); k4sm(b + 4, 2 * p + 1, s1, mn1);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(DP(lo, _mm256_loadu_si256((const __m256i *) (x.q + blk * 32)))), _mm256_set1_ps(d * s0 * x.d[blk]), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(DP(hi, _mm256_loadu_si256((const __m256i *) (x.q + blk * 32 + 32)))), _mm256_set1_ps(d * s1 * x.d[blk + 1]), acc);
            macc += dmin * (mn0 * x.s[blk] + mn1 * x.s[blk + 1]);
        }
    }
    return hsum(acc) - macc;
}
template <> float row_dot<T_Q4_K>(const uint8_t * row, int n, const Act & x) { return row_dot_k<0>(row, n, x); }
template <> float row_dot<T_Q5_K>(const uint8_t * row, int n, const Act & x) { return row_dot_k<1>(row, n, x); }

using RowDot = float (*)(const uint8_t *, int, const Act &);
static RowDot pick(int t) {
    switch (t) {
        case T_IQ2_XS: return row_dot<T_IQ2_XS>; case T_IQ3_XXS: return row_dot<T_IQ3_XXS>; case T_IQ4_XS: return row_dot<T_IQ4_XS>;
        case T_Q4_K: return row_dot<T_Q4_K>; case T_Q5_K: return row_dot<T_Q5_K>;
    }
    fprintf(stderr, "type %d not supported on the CPU\n", t); exit(1);
}

static void q8_block(const float * v, int8_t * q, float * d, float * s, int * si) {   // same rounding as g53x
    float amax = 0.f;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(v[i]));
    const float dd = amax / 127.f; int qs = 0;
    for (int i = 0; i < 32; i++) { const int qi = amax == 0.f ? 0 : (int) std::round(v[i] / dd); q[i] = (int8_t) qi; qs += qi; }
    *d = dd; *s = dd * (float) qs; *si = qs;
}

struct Job {
    const uint8_t * wg, * wu, * wd; size_t gsz, usz, dsz, grb, urb, drb; RowDot fg, fu, fd;
    const int * ids; int k; Act x;
    float * h, * gu, * out; int8_t * hq; float * hd, * hs; int * hsi;
    std::atomic<int> t1{0}, t2{0}, done1{0}, done2{0};
};
static void work(Job & j, int nthr) {
    const int n1 = j.k * (F / 32), n2 = j.k * (D / 64);
    for (int t; (t = j.t1.fetch_add(1)) < n1;) {
        const int e = t / (F / 32), r0 = t % (F / 32) * 32; const size_t slot = j.ids[e];
        float hv[32];
        for (int r = 0; r < 32; r++) {
            const float g = j.fg(j.wg + slot * j.gsz + (r0 + r) * j.grb, D, j.x), u = j.fu(j.wu + slot * j.usz + (r0 + r) * j.urb, D, j.x);
            hv[r] = g / (1.f + std::exp(-g)) * u;
            if (j.gu) { j.gu[2 * e * F + r0 + r] = g; j.gu[(2 * e + 1) * F + r0 + r] = u; }
        }
        const int bI = (e * F + r0) / 32;
        memcpy(j.h + e * F + r0, hv, sizeof hv);
        q8_block(hv, j.hq + e * F + r0, j.hd + bI, j.hs + bI, j.hsi + bI);
    }
    j.done1.fetch_add(1);
    while (j.done1.load(std::memory_order_acquire) < nthr) _mm_pause();
    for (int t; (t = j.t2.fetch_add(1)) < n2;) {
        const int e = t / (D / 64), r0 = t % (D / 64) * 64; const size_t slot = j.ids[e];
        const Act a{j.hq + e * F, j.hd + e * F / 32, j.hs + e * F / 32, j.hsi + e * F / 32};
        for (int r = 0; r < 64; r++) j.out[e * D + r0 + r] = j.fd(j.wd + slot * j.dsz + (size_t) (r0 + r) * j.drb, F, a);
    }
    j.done2.fetch_add(1);
}

static std::vector<char> slurp(const std::string & p, size_t n) {
    std::vector<char> b(n);
    FILE * f = fopen(p.c_str(), "rb");
    if (!f || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "read %s failed\n", p.c_str()); exit(1); }
    fclose(f);
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: see source\n"); return 1; }
    std::string pre = argv[1], mode = argv[6];
    const int tg = atoi(argv[2]), tu = atoi(argv[3]), td = atoi(argv[4]), nthr = atoi(argv[5]);
    const bool check = mode == "check"; const int slots = check ? NE : SLOTS;
    const size_t grb = bbytes(tg) * (D / 256), urb = bbytes(tu) * (D / 256), drb = bbytes(td) * (F / 256);
    const size_t gsz = grb * F, usz = urb * F, dsz = drb * D, rec = gsz + usz + dsz;
    auto bg = slurp(pre + "_gate.bin", gsz * NE), bu = slurp(pre + "_up.bin", usz * NE), bd = slurp(pre + "_down.bin", dsz * NE);
    uint8_t * W = (uint8_t *) aligned_alloc(64, (rec * slots + 63) / 64 * 64);
    uint8_t * wg = W, * wu = W + gsz * slots, * wd = W + (gsz + usz) * slots;
    for (int s = 0; s < slots; s++) {
        memcpy(wg + s * gsz, bg.data() + (s % NE) * gsz, gsz); memcpy(wu + s * usz, bu.data() + (s % NE) * usz, usz);
        memcpy(wd + s * dsz, bd.data() + (s % NE) * dsz, dsz);
    }
    std::vector<float> x(D), h(NE * F), out(NE * D), xd(D / 32), xs(D / 32), hd(NE * F / 32), hs(NE * F / 32), gu;
    std::vector<int8_t> xq(D), hq(NE * F); std::vector<int> xsi(D / 32), hsi(NE * F / 32), ids(16);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(rng);   // same draw as glmbench / g53x

    std::atomic<int> gen{0}; std::atomic<bool> quit{false}; Job * cur = nullptr;
    std::vector<std::thread> pool;
    for (int t = 1; t < nthr; t++) pool.emplace_back([&] {
        int seen = 0;
        while (true) {
            int g;
            while ((g = gen.load(std::memory_order_acquire)) == seen) { if (quit) return; _mm_pause(); }
            seen = g; work(*cur, nthr);
        }
    });
    auto run = [&](int k) {
        for (int b = 0; b < D / 32; b++) q8_block(&x[b * 32], &xq[b * 32], &xd[b], &xs[b], &xsi[b]);
        Job * j = new Job;
        j->wg = wg; j->wu = wu; j->wd = wd; j->gsz = gsz; j->usz = usz; j->dsz = dsz; j->grb = grb; j->urb = urb; j->drb = drb;
        j->fg = pick(tg); j->fu = pick(tu); j->fd = pick(td); j->ids = ids.data(); j->k = k;
        j->x = Act{xq.data(), xd.data(), xs.data(), xsi.data()};
        j->h = h.data(); j->gu = gu.empty() ? nullptr : gu.data(); j->out = out.data(); j->hq = hq.data(); j->hd = hd.data(); j->hs = hs.data(); j->hsi = hsi.data();
        cur = j; gen.fetch_add(1, std::memory_order_release);
        work(*j, nthr);
        while (j->done2.load(std::memory_order_acquire) < nthr) _mm_pause();
        delete j;   // every worker has left work() (done2 counts them)
    };
    if (check) {
        gu.resize(2 * NE * F);
        for (int i = 0; i < NE; i++) ids[i] = i;
        run(NE);
        std::string o = argv[7];
        auto dump = [&](const char * ext, const void * p, size_t n) { FILE * f = fopen((o + ext).c_str(), "wb"); fwrite(p, 1, n, f); fclose(f); };
        dump(".x", x.data(), D * 4); dump(".gu", gu.data(), gu.size() * 4); dump(".xq", xq.data(), D); dump(".xd", xd.data(), D / 32 * 4);
        dump(".hq", hq.data(), NE * F); dump(".hd", hd.data(), NE * F / 32 * 4); dump(".dn", out.data(), NE * D * 4); dump(".h", h.data(), NE * F * 4);
        fprintf(stderr, "wrote %s.*\n", o.c_str());
    } else {
        std::vector<int> ks;
        for (char * s = strtok(argv[7], ","); s; s = strtok(nullptr, ",")) ks.push_back(atoi(s));
        const int iters = atoi(argv[8]);
        std::vector<int> perm(slots);
        for (int k : ks) {
            std::vector<double> t;
            for (int it = 0; it < iters + 5; it++) {
                for (int i = 0; i < slots; i++) perm[i] = i;
                std::shuffle(perm.begin(), perm.end(), rng);
                auto t0 = std::chrono::steady_clock::now();
                memcpy(ids.data(), perm.data(), k * sizeof(int));
                run(k);
                auto t1 = std::chrono::steady_clock::now();
                if (it >= 5) t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(t.begin(), t.end());
            const double p50 = t[t.size() / 2], p90 = t[(size_t) (t.size() * 0.9)];
            printf("g53c %s k=%d thr=%d: p50 %.0f us p90 %.0f us  %.0f us/expert  %.1f GB/s\n", pre.substr(pre.rfind('/') + 1).c_str(), k, nthr, p50, p90, p50 / k, k * rec / p50 / 1e3);
            fflush(stdout);
        }
    }
    quit = true;
    for (auto & th : pool) th.join();
    return 0;
}
