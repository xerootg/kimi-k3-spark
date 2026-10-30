// g53x: one GLM-5.3 token's routed-expert FFN (k experts: q8_1(x), gate+up+SwiGLU, q8_1(h), down) on the Intel
// Arc iGPU, standalone SYCL (no ggml runtime). Weight blocks are ggml's, unchanged (no repack).
//
//   g53x PREFIX TG TU TD bench k1,k2,... ITERS     p50/p90 per token, weights replicated into 64 slots
//   g53x PREFIX TG TU TD check OUT                 ids 0..7, writes OUT.x / OUT.dn (glmref compares them)
// PREFIX_{gate,up,down}.bin hold 8 experts each (extract.py). Types are ggml ids:
// IQ2_XS 17, IQ3_XXS 18, IQ4_XS 23, Q4_K 12, Q5_K 13, Q6_K 14.
// Env: G53X_PROF=1 per-kernel device times; G53X_XIN=file use that x (6144 f32) instead of the seeded one.
//
// Kernel shape: 256-item work-groups = 16 sub-groups of 16 lanes, one output row per sub-group (gate and up rows
// together); the lanes stride over the row's 32-value units (IQ: one 32-block, K-quants: 64 values). The q8_1
// activation and the type's lookup tables (IQ2_XS grid 4 KiB, IQ3_XXS grid 1 KiB, sign masks 1 KiB, IQ4 nibble
// pairs 0.5 KiB) live in SLM. Integer dot products use IGC's dp4a builtin.
#include <sycl/sycl.hpp>
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int D = 6144, F = 2048, NE = 8, SLOTS = 64, SG = 16, NSG = 16;   // NSG sub-groups per work-group
#ifndef G53X_RG
#define G53X_RG 2   // gate/up rows per sub-group (each with its up row: 2*RG rows share one activation load)
#endif
#ifndef G53X_RD
#define G53X_RD 2   // down rows per sub-group
#endif
static const int RG = G53X_RG, RD = G53X_RD;
enum { T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ4_XS = 23 };

static size_t row_bytes(int t, int n) {
    switch (t) {
        case T_Q4_K: return n / 256 * 144;  case T_Q5_K: return n / 256 * 176;  case T_Q6_K: return n / 256 * 210;
        case T_IQ2_XS: return n / 256 * 74; case T_IQ3_XXS: return n / 256 * 98; case T_IQ4_XS: return n / 256 * 136;
    }
    fprintf(stderr, "type %d not supported\n", t); exit(1);
}

#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b, bool sat);
static inline int dp4a(int a, int b, int c) { return __builtin_IB_dp4a_ss(c, a, b, false); }
#else
static inline int dp4a(int a, int b, int c) {
    for (int i = 0; i < 4; i++) c += (int8_t) (a >> 8 * i) * (int8_t) (b >> 8 * i);
    return c;
}
#endif
static inline float h2f(const uint8_t * p) { uint16_t u = p[0] | (p[1] << 8); return (float) sycl::bit_cast<sycl::half>(u); }
static inline uint32_t ld16(const uint8_t * p) { return *(const uint16_t *) p; }   // 2-byte aligned in every block
static inline uint32_t ld32u(const uint8_t * p) { return ld16(p) | (ld16(p + 2) << 16); }
static inline int ld32(const uint8_t * p) { return *(const int *) p; }             // 4-byte aligned

// SLM tables (only the ones a type needs are filled)
struct Tabs { const uint64_t * g2; const uint32_t * g3; const uint64_t * sgn; const uint16_t * k4; };
static inline int apply_sign(uint32_t g, uint32_t m) { return (int) ((g ^ m) + (m & 0x01010101u)); }
#ifndef G53X_EXP
#define G53X_EXP 0
#endif
#ifndef G53X_SIGN_ALU
#define G53X_SIGN_ALU 1
#endif
// 7 stored sign bits + parity -> two 32-bit byte masks (0xff = negate), in ALU (popcount, shifts) or from SLM
static inline uint64_t sign_mask(uint32_t s7, const Tabs & t) {
#if G53X_SIGN_ALU
    const uint32_t s = (s7 & 127) | ((sycl::popcount(s7 & 127) & 1) << 7);
    auto m4 = [](uint32_t v) { const uint32_t b = (v & 1) | ((v & 2) << 7) | ((v & 4) << 14) | ((v & 8) << 21); return (b << 8) - b; };
    return (uint64_t) m4(s) | ((uint64_t) m4(s >> 4) << 32);
#else
    return t.sgn[s7 & 127];
#endif
}

// One unit's contribution. ly() loads the unit's q8_1 ints (from SLM) into registers once per unit for all rows a
// sub-group handles; dot() takes them (y) plus xd (scale) and xs (d * sum q) per 32-block.
// IQ types: unit = 32-block u of the row; K types: unit = 64 values (block pair) u; Q6_K: see there.
#include "g53x_ty.inc"

template <int T> static inline void fill_tabs(sycl::nd_item<2> it, uint64_t * g2, uint32_t * g3, uint64_t * sg, uint16_t * k4) {
    const int l = it.get_local_linear_id(), n = it.get_local_range(0) * it.get_local_range(1);
    if constexpr (T == T_IQ2_XS) { for (int i = l; i < 512; i += n) g2[i] = iq2xs_grid[i]; }
    if constexpr (T == T_IQ3_XXS) { for (int i = l; i < 256; i += n) g3[i] = iq3xxs_grid[i]; }
    if constexpr (T == T_IQ2_XS || T == T_IQ3_XXS) { for (int i = l; i < 128; i += n) sg[i] = ksigns64[i]; }
    if constexpr (T == T_IQ4_XS) {
        for (int i = l; i < 256; i += n) k4[i] = (uint16_t) ((uint8_t) kvalues_iq4nl[i & 15] | ((uint8_t) kvalues_iq4nl[i >> 4] << 8));
    }
}

// R rows of one type against the same activation: the unit's activation ints are loaded once for all R rows
template <int T, int R, int NSB> static inline void rows_dot(const uint8_t * const * rows, int n, int lane, const int * xq, const float * xd,
                                                    const float * xs, const Tabs & t, float * acc) {
    const int units = n / 256 * Ty<T>::per_sb;
    for (int r = 0; r < R; r++) acc[r] = 0.f;
#pragma unroll 2
    for (int u = lane; u < units; u += SG) {
        int y[Ty<T>::ny];
        Ty<T>::ly(u, xq, y);
#pragma unroll
        for (int r = 0; r < R; r++) acc[r] += Ty<T>::template dot<NSB>(rows[r], u, y, xd, xs, t);
    }
}

// q8_1 of r rows of n floats: one item per 32-block (amax / 127, round half away from zero). xs = d * sum(q), the
// block sum of the quantized values (ggml-sycl stores the float sum of x instead), so the K-quant min term is exact.
static void quantize(sycl::queue & q, const float * x, int8_t * xq, float * xd, float * xs, int nblk) {
    q.parallel_for(sycl::range<1>(nblk), [=](sycl::id<1> id) {
        const int bI = id[0]; const float * v = x + bI * 32;
        float amax = 0.f, sum = 0.f;
        for (int i = 0; i < 32; i++) { amax = sycl::fmax(amax, sycl::fabs(v[i])); sum += v[i]; }
        const float d = amax / 127.f;
        int qsum = 0;
        for (int i = 0; i < 32; i++) { const int qi = amax == 0.f ? 0 : (int) sycl::round(v[i] / d); xq[bI * 32 + i] = (int8_t) qi; qsum += qi; }
        xd[bI] = d; xs[bI] = d * (float) qsum; (void) sum;
    });
}

// q8_1 of nblk 32-blocks of v into SLM by the work-group's sub-groups (2 values per lane), same rounding as quantize()
static inline void q8_slm(sycl::nd_item<2> it, const float * v, int nblk, int8_t * sx, float * sd, float * ss) {
    auto sg = it.get_sub_group();
    const int lane = sg.get_local_id()[0], nsg = sg.get_group_range()[0];
    for (int b = sg.get_group_id()[0]; b < nblk; b += nsg) {
        const float a0 = v[b * 32 + lane], a1 = v[b * 32 + 16 + lane];
        const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(a0), sycl::fabs(a1)), sycl::maximum<float>());
        const float d = amax / 127.f;
        const int q0 = amax == 0.f ? 0 : (int) sycl::round(a0 / d), q1 = amax == 0.f ? 0 : (int) sycl::round(a1 / d);
        const int qs = sycl::reduce_over_group(sg, q0 + q1, sycl::plus<int>());
        sx[b * 32 + lane] = (int8_t) q0; sx[b * 32 + 16 + lane] = (int8_t) q1;
        if (lane == 0) { sd[b] = d; ss[b] = d * (float) qs; }
    }
}

// gate + up rows (same sub-group) + SwiGLU -> h[e][F]
template <int TG, int TU>
static sycl::event gateup(sycl::queue & q, const uint8_t * wg, const uint8_t * wu, const int * ids, int k,
                          const float * xf, const int8_t * xq, const float * xd, const float * xs, float * h, float * gu) {
    const size_t gsz = row_bytes(TG, D) * F, usz = row_bytes(TU, D) * F, grb = row_bytes(TG, D), urb = row_bytes(TU, D);
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int, 1> sx(D / 4, cgh);
        sycl::local_accessor<float, 1> sd(D / 32, cgh), ss(D / 32, cgh);
        sycl::local_accessor<uint64_t, 1> g2(TG == T_IQ2_XS || TU == T_IQ2_XS ? 512 : 1, cgh), sg(128, cgh);
        sycl::local_accessor<uint32_t, 1> g3(TG == T_IQ3_XXS || TU == T_IQ3_XXS ? 256 : 1, cgh);
        sycl::local_accessor<uint16_t, 1> k4(TG == T_IQ4_XS || TU == T_IQ4_XS ? 256 : 1, cgh);
        static_assert(TG == TU, "gate and up share a type in every GLM-5.3 layer");
        cgh.parallel_for(sycl::nd_range<2>({(size_t) k, (size_t) F / (NSG * RG) * SG * NSG}, {1, (size_t) SG * NSG}),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                const int l = it.get_local_id(1), e = it.get_group(0);
                if (xf) q8_slm(it, xf, D / 32, (int8_t *) &sx[0], &sd[0], &ss[0]);   // fused q8_1(x)
                else {
                    for (int i = l; i < D / 4; i += SG * NSG) sx[i] = ((const int *) xq)[i];
                    for (int i = l; i < D / 32; i += SG * NSG) { sd[i] = xd[i]; ss[i] = xs[i]; }
                }
                uint64_t * pg2 = &g2[0], * psg = &sg[0]; uint32_t * pg3 = &g3[0]; uint16_t * pk4 = &k4[0];
                fill_tabs<TG>(it, pg2, pg3, psg, pk4);
                if constexpr (TU != TG) fill_tabs<TU>(it, pg2, pg3, psg, pk4);
                sycl::group_barrier(it.get_group());
                const Tabs t{pg2, pg3, psg, pk4};
                auto sgp = it.get_sub_group();
                const int lane = sgp.get_local_id()[0], j0 = (it.get_group(1) * NSG + sgp.get_group_id()[0]) * RG;
                const size_t slot = ids[e];
                const uint8_t * rows[2 * RG];
                for (int r = 0; r < RG; r++) { rows[r] = wg + slot * gsz + (j0 + r) * grb; rows[RG + r] = wu + slot * usz + (j0 + r) * urb; }
                float acc[2 * RG];
                rows_dot<TG, 2 * RG, D / 256>(rows, D, lane, &sx[0], &sd[0], &ss[0], t, acc);
                for (int r = 0; r < RG; r++) {
                    const float g = sycl::reduce_over_group(sgp, acc[r], sycl::plus<float>());
                    const float u = sycl::reduce_over_group(sgp, acc[RG + r], sycl::plus<float>());
                    if (lane == 0) {
                        h[e * F + j0 + r] = g / (1.f + sycl::exp(-g)) * u;
                        if (gu) { gu[(2 * e) * F + j0 + r] = g; gu[(2 * e + 1) * F + j0 + r] = u; }
                    }
                }
            });
    });
}

template <int TD>
static sycl::event down(sycl::queue & q, const uint8_t * wd, const int * ids, int k, const float * hf, const int8_t * hq, const float * hd,
                        const float * hs, float * out) {
    const size_t dsz = row_bytes(TD, F) * D, drb = row_bytes(TD, F);
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int, 1> sx(F / 4, cgh);
        sycl::local_accessor<float, 1> sd(F / 32, cgh), ss(F / 32, cgh);
        sycl::local_accessor<uint64_t, 1> g2(TD == T_IQ2_XS ? 512 : 1, cgh), sg(128, cgh);
        sycl::local_accessor<uint32_t, 1> g3(TD == T_IQ3_XXS ? 256 : 1, cgh);
        sycl::local_accessor<uint16_t, 1> k4(TD == T_IQ4_XS ? 256 : 1, cgh);
        cgh.parallel_for(sycl::nd_range<2>({(size_t) k, (size_t) D / (NSG * RD) * SG * NSG}, {1, (size_t) SG * NSG}),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                const int l = it.get_local_id(1), e = it.get_group(0);
                if (hf) q8_slm(it, hf + e * F, F / 32, (int8_t *) &sx[0], &sd[0], &ss[0]);   // fused q8_1(h)
                else {
                    for (int i = l; i < F / 4; i += SG * NSG) sx[i] = ((const int *) hq)[e * F / 4 + i];
                    for (int i = l; i < F / 32; i += SG * NSG) { sd[i] = hd[e * F / 32 + i]; ss[i] = hs[e * F / 32 + i]; }
                }
                uint64_t * pg2 = &g2[0], * psg = &sg[0]; uint32_t * pg3 = &g3[0]; uint16_t * pk4 = &k4[0];
                fill_tabs<TD>(it, pg2, pg3, psg, pk4);
                sycl::group_barrier(it.get_group());
                const Tabs t{pg2, pg3, psg, pk4};
                auto sgp = it.get_sub_group();
                const int lane = sgp.get_local_id()[0], r0 = (it.get_group(1) * NSG + sgp.get_group_id()[0]) * RD;
                const uint8_t * rows[RD];
                for (int r = 0; r < RD; r++) rows[r] = wd + (size_t) ids[e] * dsz + (size_t) (r0 + r) * drb;
                float acc[RD];
                rows_dot<TD, RD, F / 256>(rows, F, lane, &sx[0], &sd[0], &ss[0], t, acc);
                for (int r = 0; r < RD; r++) {
                    const float s = sycl::reduce_over_group(sgp, acc[r], sycl::plus<float>());
                    if (lane == 0) out[e * D + r0 + r] = s;
                }
            });
    });
}

struct Bufs { float * gu = nullptr; int8_t * xq; float * xd, * xs, * h; int8_t * hq; float * hd, * hs, * out; float * x; int * ids; };
using GateUpFn = sycl::event (*)(sycl::queue &, const uint8_t *, const uint8_t *, const int *, int, const float *, const int8_t *, const float *, const float *, float *, float *);
using DownFn = sycl::event (*)(sycl::queue &, const uint8_t *, const int *, int, const float *, const int8_t *, const float *, const float *, float *);

static GateUpFn pick_gu(int tg, int tu) {
#define GU(a, b) if (tg == a && tu == b) return gateup<a, b>;
    GU(T_IQ2_XS, T_IQ2_XS) GU(T_IQ3_XXS, T_IQ3_XXS) GU(T_Q4_K, T_Q4_K) GU(T_Q5_K, T_Q5_K)
#undef GU
    fprintf(stderr, "gate/up %d/%d not instantiated\n", tg, tu); exit(1);
}
static DownFn pick_dn(int td) {
    switch (td) {
        case T_IQ3_XXS: return down<T_IQ3_XXS>; case T_IQ4_XS: return down<T_IQ4_XS>;
        case T_Q5_K: return down<T_Q5_K>; case T_Q6_K: return down<T_Q6_K>;
    }
    fprintf(stderr, "down %d not instantiated\n", td); exit(1);
}

static std::vector<char> slurp(const std::string & p, size_t n) {
    std::vector<char> b(n);
    FILE * f = fopen(p.c_str(), "rb");
    if (!f || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "read %s failed\n", p.c_str()); exit(1); }
    fclose(f);
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: see source\n"); return 1; }
    std::string pre = argv[1], mode = argv[5];
    const int tg = atoi(argv[2]), tu = atoi(argv[3]), td = atoi(argv[4]);
    const bool check = mode == "check";
    const int slots = check ? NE : SLOTS;
    const bool prof = getenv("G53X_PROF") && atoi(getenv("G53X_PROF"));
    sycl::property_list pl = prof ? sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}
                                  : sycl::property_list{sycl::property::queue::in_order()};
    setenv("UR_L0_USE_IMMEDIATE_COMMANDLISTS", "1", 0);   // batched lists delay submission (k3x found the same)
    sycl::queue q{sycl::gpu_selector_v, pl};
    fprintf(stderr, "device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const size_t gsz = row_bytes(tg, D) * F, usz = row_bytes(tu, D) * F, dsz = row_bytes(td, F) * D, rec = gsz + usz + dsz;
    auto bg = slurp(pre + "_gate.bin", gsz * NE), bu = slurp(pre + "_up.bin", usz * NE), bd = slurp(pre + "_down.bin", dsz * NE);
    uint8_t * wg = sycl::malloc_device<uint8_t>(gsz * slots, q), * wu = sycl::malloc_device<uint8_t>(usz * slots, q),
            * wd = sycl::malloc_device<uint8_t>(dsz * slots, q);
    {   // repack every expert's rows into the field-run layout (same size)
        std::vector<char> t(std::max(gsz, std::max(usz, dsz)) * NE);
        repack(tg, (const uint8_t *) bg.data(), (uint8_t *) t.data(), (size_t) F * NE, D); memcpy(bg.data(), t.data(), gsz * NE);
        repack(tu, (const uint8_t *) bu.data(), (uint8_t *) t.data(), (size_t) F * NE, D); memcpy(bu.data(), t.data(), usz * NE);
        repack(td, (const uint8_t *) bd.data(), (uint8_t *) t.data(), (size_t) D * NE, F); memcpy(bd.data(), t.data(), dsz * NE);
    }
    for (int s = 0; s < slots; s++) {
        q.memcpy(wg + s * gsz, bg.data() + (s % NE) * gsz, gsz); q.memcpy(wu + s * usz, bu.data() + (s % NE) * usz, usz);
        q.memcpy(wd + s * dsz, bd.data() + (s % NE) * dsz, dsz);
    }
    q.wait();
    Bufs B;
    B.x = sycl::malloc_host<float>(D, q); B.ids = sycl::malloc_host<int>(16, q); B.out = sycl::malloc_host<float>(NE * D, q);
    B.xq = sycl::malloc_device<int8_t>(D, q); B.xd = sycl::malloc_device<float>(D / 32, q); B.xs = sycl::malloc_device<float>(D / 32, q);
    B.h = sycl::malloc_device<float>(NE * F, q); B.hq = sycl::malloc_device<int8_t>(NE * F, q);
    B.hd = sycl::malloc_device<float>(NE * F / 32, q); B.hs = sycl::malloc_device<float>(NE * F / 32, q);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < D; i++) B.x[i] = nd(rng);   // same draw as glmbench
    if (getenv("G53X_XIN")) { auto xb = slurp(getenv("G53X_XIN"), D * 4); memcpy(B.x, xb.data(), D * 4); }

    GateUpFn gu = pick_gu(tg, tu); DownFn dn = pick_dn(td);
    std::vector<sycl::event> ev(4);
    const bool fuseq = !getenv("G53X_FUSEQ") || atoi(getenv("G53X_FUSEQ"));   // check mode runs both and compares
    auto run = [&](int k) {
        if (fuseq) {   // two launches: q8_1 inside the matvec kernels
            ev[1] = gu(q, wg, wu, B.ids, k, B.x, nullptr, nullptr, nullptr, B.h, B.gu);
            ev[3] = dn(q, wd, B.ids, k, B.h, nullptr, nullptr, nullptr, B.out);
        } else {
            quantize(q, B.x, B.xq, B.xd, B.xs, D / 32);
            ev[1] = gu(q, wg, wu, B.ids, k, nullptr, B.xq, B.xd, B.xs, B.h, B.gu);
            quantize(q, B.h, B.hq, B.hd, B.hs, k * F / 32);
            ev[3] = dn(q, wd, B.ids, k, nullptr, B.hq, B.hd, B.hs, B.out);
        }
        ev[3].wait();
    };

    if (check) {
        for (int i = 0; i < NE; i++) B.ids[i] = i;
        B.gu = sycl::malloc_shared<float>(2 * NE * F, q);
        std::vector<float> fused;
        if (fuseq) { run(NE); fused.assign(B.out, B.out + NE * D); }
        const bool f0 = fuseq; const_cast<bool &>(fuseq) = false; run(NE); const_cast<bool &>(fuseq) = f0;
        if (f0) fprintf(stderr, "fused q8_1 vs separate quantize kernels: dn %s\n", memcmp(fused.data(), B.out, NE * D * 4) ? "DIFFER" : "bit-identical");
        auto dump = [&](const std::string & name, const void * dev, size_t n) {
            std::vector<char> v(n); q.memcpy(v.data(), dev, n).wait();
            FILE * f = fopen((std::string(argv[6]) + name).c_str(), "wb"); fwrite(v.data(), 1, n, f); fclose(f);
        };
        dump(".gu", B.gu, 2 * NE * F * 4); dump(".xq", B.xq, D); dump(".xd", B.xd, D / 32 * 4);
        dump(".hq", B.hq, NE * F); dump(".hd", B.hd, NE * F / 32 * 4);
        std::string o = argv[6];
        FILE * f = fopen((o + ".x").c_str(), "wb"); fwrite(B.x, 4, D, f); fclose(f);
        f = fopen((o + ".dn").c_str(), "wb"); fwrite(B.out, 4, NE * D, f); fclose(f);
        std::vector<float> h(NE * F); q.memcpy(h.data(), B.h, NE * F * 4).wait();
        f = fopen((o + ".h").c_str(), "wb"); fwrite(h.data(), 4, NE * F, f); fclose(f);
        fprintf(stderr, "wrote %s.{x,h,dn}\n", o.c_str());
        return 0;
    }
    std::vector<int> ks;
    for (char * s = strtok(argv[6], ","); s; s = strtok(nullptr, ",")) ks.push_back(atoi(s));
    const int iters = atoi(argv[7]);
    std::vector<int> perm(slots);
    for (int k : ks) {
        std::vector<double> t, tgu, tdn;
        for (int it = 0; it < iters + 5; it++) {
            for (int i = 0; i < slots; i++) perm[i] = i;
            std::shuffle(perm.begin(), perm.end(), rng);
            auto t0 = std::chrono::steady_clock::now();
            memcpy(B.ids, perm.data(), k * sizeof(int));
            run(k);
            auto t1 = std::chrono::steady_clock::now();
            if (it < 5) continue;
            t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            if (prof) {
                auto dt = [](sycl::event & e) { return (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                                        e.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e3; };
                tgu.push_back(dt(ev[1])); tdn.push_back(dt(ev[3]));
            }
        }
        auto med = [](std::vector<double> & v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
        double p50 = med(t), p90 = t[(size_t) (t.size() * 0.9)];
        printf("g53x %s k=%d: p50 %.0f us p90 %.0f us  %.0f us/expert  %.1f GB/s", pre.substr(pre.rfind('/') + 1).c_str(), k, p50, p90, p50 / k, k * rec / p50 / 1e3);
        if (prof) printf("  [gate/up %.0f us %.1f GB/s, down %.0f us %.1f GB/s]", med(tgu), k * (gsz + usz) / med(tgu) / 1e3, med(tdn), k * dsz / med(tdn) / 1e3);
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
