// g53x: one GLM-5.3 token's routed-expert FFN (k experts: q8_1(x), gate+up+SwiGLU, q8_1(h), down) on the Intel
// Arc iGPU, standalone SYCL (no ggml runtime). Weight blocks are ggml's, unchanged (no repack).
//
//   g53x PREFIX TG TU TD bench k1,k2,... ITERS     p50/p90 per token, weights replicated into 64 slots
//   g53x PREFIX TG TU TD check OUT                 ids 0..7, writes OUT.x / OUT.dn (glmref compares them)
//   g53x SET STORE HIDDEN - serveidx PORT          serve a K3XS expert store with k3x's serveidx protocol (serve_index)
// PREFIX_{gate,up,down}.bin hold 8 experts each (extract.py). Types are ggml ids:
// IQ2_XS 17, IQ3_XXS 18, IQ4_XS 23, Q4_K 12, Q5_K 13, Q6_K 14.
// Env: G53X_PROF=1 per-kernel device times; G53X_XIN=file use that x (6144 f32) instead of the seeded one;
// G53X_HOSTW=1 weights in host USM instead of device USM.
//
// Kernel shape: 256-item work-groups = 16 sub-groups of 16 lanes, one output row per sub-group (gate and up rows
// together); the lanes stride over the row's 32-value units (IQ: one 32-block, K-quants: 64 values). The q8_1
// activation and the type's lookup tables (IQ2_XS grid 4 KiB, IQ3_XXS grid 1 KiB, sign masks 1 KiB, IQ4 nibble
// pairs 0.5 KiB) live in SLM. Integer dot products use IGC's dp4a builtin.
#include <sycl/sycl.hpp>
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include "expert_disk.h"   // tools/bitexact: K3XS store reader (serveidx)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static const int D = 6144, F = 2048, NE = 8, SLOTS = 64, SG = 16, NSG = 16;   // NSG sub-groups per work-group
#ifndef G53X_RG
#define G53X_RG 2   // gate/up rows per sub-group (each with its up row: 2*RG rows share one activation load)
#endif
#ifndef G53X_RD
#define G53X_RD 2   // down rows per sub-group
#endif
static const int RG = G53X_RG, RD = G53X_RD;
enum { T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ4_XS = 23 };

static size_t type_rb(int t, long n) {   // bytes of one row of n values; 0: type not supported
    switch (t) {
        case T_Q4_K: return n / 256 * 144;  case T_Q5_K: return n / 256 * 176;  case T_Q6_K: return n / 256 * 210;
        case T_IQ2_XS: return n / 256 * 74; case T_IQ3_XXS: return n / 256 * 98; case T_IQ4_XS: return n / 256 * 136;
    }
    return 0;
}
static size_t row_bytes(int t, int n) {
    if (const size_t b = type_rb(t, n)) return b;
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

// The weights of expert e (e = 0..k-1 of a token) are W[3e] (gate), W[3e + 1] (up), W[3e + 2] (down): field-run
// matrices (repack) anywhere in device or host USM. D (hidden size) and F (expert FFN size) are multiples of 256, template
// parameters (compile-time field offsets: a runtime row length costs ~1.5x) that shadow the bench/check constants.
// G53X_DIMS lists the instantiated (D, F) pairs; serveidx takes them from the store and rejects others.
// A pointer loaded from memory is generic address space; tell the compiler it is global (USM), as a captured base is
// (generic loads cost ~1.5x here).
static inline const uint8_t * gptr(const uint8_t * p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p).get();
}
// gate + up rows (same sub-group) + SwiGLU -> h[e][F]
template <int TG, int TU, int D, int F>
static sycl::event gateup(sycl::queue & q, const uint8_t * const * W, int k,
                          const float * xf, const int8_t * xq, const float * xd, const float * xs, float * h, float * gu) {
    const size_t grb = row_bytes(TG, D), urb = row_bytes(TU, D);
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
                const uint8_t * wg = gptr(W[3 * e]), * wu = gptr(W[3 * e + 1]);
                const uint8_t * rows[2 * RG];
                for (int r = 0; r < RG; r++) { rows[r] = wg + (size_t) (j0 + r) * grb; rows[RG + r] = wu + (size_t) (j0 + r) * urb; }
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

template <int TD, int D, int F>
static sycl::event down(sycl::queue & q, const uint8_t * const * W, int k, const float * hf, const int8_t * hq, const float * hd,
                        const float * hs, float * out) {
    const size_t drb = row_bytes(TD, F);
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
                const uint8_t * wd = gptr(W[3 * e + 2]);
                for (int r = 0; r < RD; r++) rows[r] = wd + (size_t) (r0 + r) * drb;
                float acc[RD];
                rows_dot<TD, RD, F / 256>(rows, F, lane, &sx[0], &sd[0], &ss[0], t, acc);
                for (int r = 0; r < RD; r++) {
                    const float s = sycl::reduce_over_group(sgp, acc[r], sycl::plus<float>());
                    if (lane == 0) out[e * D + r0 + r] = s;
                }
            });
    });
}

struct Bufs { float * gu = nullptr; int8_t * xq; float * xd, * xs, * h; int8_t * hq; float * hd, * hs, * out; float * x; const uint8_t ** W; };
using GateUpFn = sycl::event (*)(sycl::queue &, const uint8_t * const *, int, const float *, const int8_t *, const float *, const float *, float *, float *);
using DownFn = sycl::event (*)(sycl::queue &, const uint8_t * const *, int, const float *, const int8_t *, const float *, const float *, float *);

#ifndef G53X_DIMS
#define G53X_DIMS DIM(6144, 2048)   // GLM-5.3: hidden 6144, expert FFN 2048
#endif
template <int DD, int FF> static GateUpFn pick_gu_df(int tg, int tu) {
#define GU(a, b) if (tg == a && tu == b) return gateup<a, b, DD, FF>;
    GU(T_IQ2_XS, T_IQ2_XS) GU(T_IQ3_XXS, T_IQ3_XXS) GU(T_Q4_K, T_Q4_K) GU(T_Q5_K, T_Q5_K)
#undef GU
    return nullptr;
}
template <int DD, int FF> static DownFn pick_dn_df(int td) {
    switch (td) {
        case T_IQ3_XXS: return down<T_IQ3_XXS, DD, FF>; case T_IQ4_XS: return down<T_IQ4_XS, DD, FF>;
        case T_Q5_K: return down<T_Q5_K, DD, FF>; case T_Q6_K: return down<T_Q6_K, DD, FF>;
    }
    return nullptr;
}
static GateUpFn pick_gu(int tg, int tu, int d, int f) {   // nullptr: not instantiated
#define DIM(a, b) if (d == a && f == b) return pick_gu_df<a, b>(tg, tu);
    G53X_DIMS
#undef DIM
    return nullptr;
}
static DownFn pick_dn(int td, int d, int f) {
#define DIM(a, b) if (d == a && f == b) return pick_dn_df<a, b>(td);
    G53X_DIMS
#undef DIM
    return nullptr;
}

static std::vector<char> slurp(const std::string & p, size_t n) {
    std::vector<char> b(n);
    FILE * f = fopen(p.c_str(), "rb");
    if (!f || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "read %s failed\n", p.c_str()); exit(1); }
    fclose(f);
    return b;
}

// ---- serveidx: the k3x serveidx wire protocol over a K3XS expert store ----
// Request: int32 {k, gid_1..gid_k, 0.., hdr[16] = seq when k < 16} (17 ints), then D float32 x. Reply: k rows of D float32
// in request order (streamed: each row is sent as soon as its down kernel is done). k outside 1..16 closes the
// connection. gid = layer * n_expert + expert (n_expert from the store header). If any id is not served, or the ids are of
// different kinds (types / sizes), the reply is k rows of NaN (the engine computes NaN rows itself). An all-zero row
// (never a real expert row; a dropped GPU submission) is sent as NaN too and counted.
static long g_rx_spin_ns = -1;
static uint64_t mono_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t) t.tv_sec * 1000000000ull + t.tv_nsec; }
static bool rd_spin(int fd, void * p, size_t n) {   // busy-poll; after K3X_RX_SPIN_US idle block in poll()
    char * c = (char *) p; uint64_t t_idle = 0;
    while (n) {
        const ssize_t r = recv(fd, c, n, MSG_DONTWAIT);
        if (r == 0) return false;
        if (r < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return false;
            if (g_rx_spin_ns >= 0) {
                const uint64_t now = mono_ns();
                if (!t_idle) t_idle = now;
                else if (now - t_idle > (uint64_t) g_rx_spin_ns) {
                    pollfd pf{fd, POLLIN, 0};
                    if (poll(&pf, 1, 100) < 0 && errno != EINTR) return false;
                    t_idle = 0;
                }
            }
            continue;
        }
        c += r; n -= r;
    }
    return true;
}
static bool wr_all(int fd, const void * p, size_t n) {
    const char * c = (const char *) p;
    while (n) { const ssize_t r = send(fd, c, n, MSG_NOSIGNAL); if (r <= 0) return false; c += r; n -= r; }
    return true;
}
// K3X_MEMWARM=<MB/s> (K3X_MEMWARM_CPU=<cpu>): stream a 64 MiB buffer at about that rate while waiting for a request,
// so the memory clocks do not drop between requests (as in k3x).
static std::atomic<int> g_warm_on{0};
static void memwarm_start(double mbps) {
    if (mbps <= 0) return;
    std::thread([mbps]() {
        { sched_param sp{}; pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp); }
        if (const char * mc = getenv("K3X_MEMWARM_CPU")) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(atoi(mc), &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs); }
        const size_t N = (size_t) 64 << 20, step = 64 << 10; uint8_t * b = (uint8_t *) aligned_alloc(4096, N); memset(b, 1, N);
        size_t off = 0; volatile uint64_t sink = 0;
        for (;;) {
            if (!g_warm_on.load(std::memory_order_relaxed)) { usleep(50); continue; }
            const uint64_t t0 = mono_ns(); uint64_t acc = 0;
            for (size_t i = 0; i < step; i += 64) acc += *(const uint64_t *) (b + off + i);
            sink = acc; off = (off + step) % N;
            while ((mono_ns() - t0) / 1e3 < step / mbps && g_warm_on.load(std::memory_order_relaxed)) {}
        }
    }).detach();
    fprintf(stderr, "K3X_MEMWARM: %.0f MB/s while idle\n", mbps);
}

// "dims" of every layer's gate tensor ([hidden, ffn, n_expert]) from the store's .json sidecar (repack-experts.py)
static std::unordered_map<long, std::pair<long, long>> sidecar_dims(const std::string & path) {
    std::unordered_map<long, std::pair<long, long>> m;
    FILE * f = fopen(path.c_str(), "rb"); if (!f) return m;
    std::string j; char b[65536]; size_t n; while ((n = fread(b, 1, sizeof b, f)) > 0) j.append(b, n); fclose(f);
    for (size_t p = j.find("\"layer\":"); p != std::string::npos; p = j.find("\"layer\":", p + 1)) {
        const long layer = strtol(j.c_str() + p + 8, nullptr, 10);
        const size_t g = j.find("\"gate\":", p), d = g == std::string::npos ? g : j.find("\"dims\":", g), lb = d == std::string::npos ? d : j.find('[', d);
        if (lb == std::string::npos) break;
        char * e; const long a = strtol(j.c_str() + lb + 1, &e, 10); while (*e == ',' || *e == ' ' || *e == '\n') e++;
        m[layer] = {a, strtol(e, nullptr, 10)};
    }
    return m;
}

// g53x SET STORE HIDDEN - serveidx PORT
//   STORE   K3XS expert store (tools/repack-experts.py, v1 or v2), read with tools/bitexact/expert_disk.h
//   SET     the experts to serve: "layer expert" lines, or k3x IDX lines "gid tg td offset" (types and offsets are
//           checked against the store), or "all" / "-" for every record of the store
//   HIDDEN  hidden size, or "-": per layer from STORE.json ("dims" of the gate tensor). The FFN size and the three
//           tensor types of every layer come from the store index (FFN = gate bytes / gate row bytes).
// Every served expert is read once, repacked to the field-run layout, into host-USM chunks of 1 GiB (the arena); the
// request path does no allocation. Env (k3x names): K3X_MAX_GB (24), K3X_SETOUT, K3X_KEEPCACHE, K3X_NOSTREAM,
// K3X_RT, K3X_RX_SPIN_US, K3X_MEMWARM, K3X_MEMWARM_CPU, K3X_FORCEWAKE.
static int serve_index(const char * setp, const char * storep, const char * hidp, int port) {
    std::unique_ptr<k3disk::Store> store;
    try { store.reset(new k3disk::Store(storep)); } catch (const std::exception & e) { fprintf(stderr, "%s: %s\n", storep, e.what()); return 1; }
    const long NX = store->experts;
    // per layer: hidden (arg or sidecar), FFN from the index, validated against the instantiated kernels
    std::unordered_map<long, std::pair<long, long>> side;
    long hid_arg = 0;
    if (!strcmp(hidp, "-")) {
        side = sidecar_dims(std::string(storep) + ".json");
        if (side.empty()) { fprintf(stderr, "HIDDEN '-' needs %s.json (repack-experts.py writes it)\n", storep); return 1; }
    } else if ((hid_arg = atol(hidp)) <= 0) { fprintf(stderr, "bad HIDDEN %s\n", hidp); return 1; }
    struct Kind { int tg, tu, td, D, F; GateUpFn gu; DownFn dn; size_t gsz, usz, dsz; };
    std::vector<Kind> kinds; kinds.reserve(64);
    std::unordered_map<long, int> layer_kind;   // layer -> kind index, -1 unsupported
    int Dw = 0;                                  // the wire's hidden size (one per server)
    for (size_t i = 0; i < store->records.size(); i += NX) {
        const k3disk::Record & r = store->records[i];
        const long layer = (long) (r.gid / NX);
        const long D = hid_arg ? hid_arg : (side.count(layer) ? side[layer].first : 0);
        char why[160] = "";
        long F = 0; size_t grb = 0, urb = 0, drb = 0;
        if (D <= 0) snprintf(why, sizeof why, "hidden size unknown (not in the sidecar)");
        else if (D % 256) snprintf(why, sizeof why, "hidden %ld not a multiple of 256", D);
        else if (!(grb = type_rb(r.tg, D)) || !type_rb(r.tu, D)) snprintf(why, sizeof why, "gate/up type %u/%u not supported", r.tg, r.tu);
        else if (r.gate % grb) snprintf(why, sizeof why, "gate bytes %llu not whole rows of %zu", (unsigned long long) r.gate, grb);
        else if ((F = (long) (r.gate / grb)) % 256 || !(drb = type_rb(r.td, F))) snprintf(why, sizeof why, "ffn %ld / down type %u not supported", F, r.td);
        else if (r.up != (urb = type_rb(r.tu, D)) * F || r.down != drb * D) snprintf(why, sizeof why, "up/down bytes do not match hidden %ld ffn %ld", D, F);
        else if (!side.empty() && side[layer].second != F) snprintf(why, sizeof why, "ffn %ld from the index, %ld in the sidecar", F, side[layer].second);
        else if (grb % 16 || urb % 16 || drb % 16) snprintf(why, sizeof why, "row bytes not 16-byte aligned");
        else if (Dw && D != Dw) snprintf(why, sizeof why, "hidden %ld differs from the other layers' %d", D, Dw);
        int ki = -1;
        if (!*why) {
            for (size_t k = 0; k < kinds.size(); k++) { const Kind & K = kinds[k];
                if (K.tg == (int) r.tg && K.tu == (int) r.tu && K.td == (int) r.td && K.D == D && K.F == F) ki = (int) k; }
            if (ki < 0) {
                GateUpFn gu = pick_gu(r.tg, r.tu, (int) D, (int) F); DownFn dn = pick_dn(r.td, (int) D, (int) F);
                if (!gu || !dn) snprintf(why, sizeof why, "types %u/%u/%u at %ldx%ld not instantiated (G53X_DIMS)", r.tg, r.tu, r.td, D, F);
                else { kinds.push_back({(int) r.tg, (int) r.tu, (int) r.td, (int) D, (int) F, gu, dn, r.gate, r.up, r.down}); ki = (int) kinds.size() - 1; }
            }
        }
        if (*why) fprintf(stderr, "layer %ld skipped (its ids answer NaN): %s\n", layer, why);
        else Dw = (int) D;
        layer_kind[layer] = ki;
    }
    if (!Dw) { fprintf(stderr, "no servable layer in %s\n", storep); return 1; }
    // the set: gids in file order
    std::vector<long> want;
    if (!strcmp(setp, "-") || !strcmp(setp, "all")) { for (auto & r : store->records) want.push_back((long) r.gid); }
    else {
        FILE * f = fopen(setp, "r"); if (!f) { perror(setp); return 1; }
        char line[256]; long bad = 0;
        while (fgets(line, sizeof line, f)) {
            long a[4]; const int n = sscanf(line, "%ld %ld %ld %ld", &a[0], &a[1], &a[2], &a[3]);
            if (n == 2) { if (a[1] < 0 || a[1] >= NX) { bad++; continue; } want.push_back(a[0] * NX + a[1]); }
            else if (n == 4) {   // k3x IDX "gid tg td offset": must describe the store's record
                const k3disk::Record * r = store->find((uint64_t) a[0]);
                if (!r || (long) r->tg != a[1] || (long) r->td != a[2] || (long) r->offset != a[3]) { fprintf(stderr, "IDX line does not match the store: %s", line); return 1; }
                want.push_back(a[0]);
            } else if (n > 0) { fprintf(stderr, "bad SET line: %s", line); return 1; }
        }
        fclose(f);
        if (bad) fprintf(stderr, "%ld SET lines with expert >= %ld ignored\n", bad, NX);
    }
    // load: one read + repack per record, straight into host USM
    setenv("UR_L0_USE_IMMEDIATE_COMMANDLISTS", "1", 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    fprintf(stderr, "device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    struct Served { int kind; const uint8_t * p; };
    std::unordered_map<long, Served> map; map.reserve(want.size() * 2 + 16);
    const double max_gb = getenv("K3X_MAX_GB") ? atof(getenv("K3X_MAX_GB")) : 24.0;
    const size_t CH = (size_t) 1 << 30; uint8_t * chunk = nullptr; size_t used = CH, tot = 0, maxrec = 0; long missing = 0;
    for (auto & K : kinds) maxrec = std::max(maxrec, K.gsz + K.usz + K.dsz);
    std::vector<uint8_t> stage(maxrec);
    int fd = open(storep, O_RDONLY); if (fd < 0) { perror(storep); return 1; }
    const double t0 = mono_ns() / 1e6;
    for (long gid : want) {
        const k3disk::Record * r = store->find((uint64_t) gid);
        if (!r) { missing++; continue; }
        const int ki = layer_kind[gid / NX];
        if (ki < 0 || map.count(gid)) continue;
        const Kind & K = kinds[ki]; const size_t prec = K.gsz + K.usz + K.dsz, pad = (prec + 4095) & ~(size_t) 4095;
        if ((tot + prec) / 1e9 > max_gb) { fprintf(stderr, "K3X_MAX_GB %.1f reached: serving %zu of %zu records\n", max_gb, map.size(), want.size()); break; }
        if (prec > CH) { fprintf(stderr, "record of %zu bytes exceeds a 1 GiB chunk\n", prec); return 1; }
        if (used + prec > CH) {
            chunk = sycl::malloc_host<uint8_t>(CH, q);
            if (!chunk) { fprintf(stderr, "host USM alloc failed after %.1f GB\n", tot / 1e9); return 1; }
            used = 0;
        }
        for (size_t got = 0; got < prec;) {
            const ssize_t n = pread(fd, stage.data() + got, prec - got, (off_t) (r->offset + got));
            if (n <= 0) { fprintf(stderr, "short read gid %ld\n", gid); return 1; }
            got += n;
        }
        if (!getenv("K3X_KEEPCACHE")) posix_fadvise(fd, (off_t) r->offset, (off_t) r->bytes, POSIX_FADV_DONTNEED);
        uint8_t * p = chunk + used;
        repack(K.tg, stage.data(), p, K.F, K.D);
        repack(K.tu, stage.data() + K.gsz, p + K.gsz, K.F, K.D);
        repack(K.td, stage.data() + K.gsz + K.usz, p + K.gsz + K.usz, K.D, K.F);
        map[gid] = {ki, p}; used += pad; tot += prec;
    }
    close(fd);
    std::vector<uint8_t>().swap(stage);
    if (missing) fprintf(stderr, "%ld SET ids are not in the store\n", missing);
    fprintf(stderr, "loaded %zu of %zu records (%.1f GB, %s arena) in %.1f s, %zu engine kinds\n", map.size(), want.size(), tot / 1e9,
            "host USM", (mono_ns() / 1e6 - t0) / 1e3, kinds.size());
    if (const char * so = getenv("K3X_SETOUT")) {   // "layer expert" of the served records (LLAMA_EXPERT_STORE_REMOTE_SET)
        FILE * f = fopen(so, "w"); if (!f) { perror(so); return 1; }
        for (long gid : want) if (map.count(gid)) fprintf(f, "%ld %ld\n", gid / NX, gid % NX);
        fclose(f);
    }
    // the request slot (one request at a time), all allocated here: header + x received straight into host USM
    int Fmax = 0; for (auto & K : kinds) Fmax = std::max(Fmax, K.F);
    const int D = Dw;
    uint8_t * sb = sycl::malloc_host<uint8_t>(4096 + (size_t) D * 4, q);   // [60 B pad][hdr 68 B][x at +128]
    int32_t * hdr = (int32_t *) (sb + 60); float * x = (float *) (sb + 128);
    float * rows = sycl::malloc_host<float>((size_t) 16 * D, q);
    const uint8_t ** W = sycl::malloc_host<const uint8_t *>(3 * 16, q);
    float * h = sycl::malloc_device<float>((size_t) 16 * Fmax, q);
    std::vector<float> nanrows((size_t) 16 * D, NAN);
    sycl::event ev[16];
    const bool stream = !getenv("K3X_NOSTREAM");
    // gate/up for all k in one launch; down per expert when streaming (rows go out as they finish), else one launch
    auto run = [&](const Kind & K, int k) {
        K.gu(q, W, k, x, nullptr, nullptr, nullptr, h, nullptr);
        if (stream) for (int j = 0; j < k; j++) ev[j] = K.dn(q, W + 3 * j, 1, h + (size_t) j * K.F, nullptr, nullptr, nullptr, rows + (size_t) j * D);
        else ev[k - 1] = K.dn(q, W, k, h, nullptr, nullptr, nullptr, rows);
    };
    const bool blockw = getenv("G53X_WAIT") && atoi(getenv("G53X_WAIT"));
    auto done = [&](int j) { if (blockw) { ev[j].wait(); return; } while (ev[j].get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {} };
    auto set_w = [&](int j, const Kind & K, const uint8_t * p) { W[3 * j] = p; W[3 * j + 1] = p + K.gsz; W[3 * j + 2] = p + K.gsz + K.usz; };
    // warm-up + self-check: the first and the last record of every kind must give a non-zero row
    for (size_t ki = 0; ki < kinds.size(); ki++) {
        const uint8_t * first = nullptr, * last = nullptr;
        for (auto & kv : map) if (kv.second.kind == (int) ki) {
            if (!first || kv.second.p < first) first = kv.second.p;
            if (!last || kv.second.p > last) last = kv.second.p;
        }
        for (const uint8_t * probe : {first, last}) {
            if (!probe) continue;
            for (int i = 0; i < D; i++) x[i] = 0.05f * (float) ((i * 37) % 41 - 20);
            set_w(0, kinds[ki], probe); run(kinds[ki], 1); q.wait();
            int nz = 0; for (int i = 0; i < D; i++) nz += rows[i] != 0.0f;
            if (!nz) { fprintf(stderr, "SELF-CHECK FAILED: all-zero row (GPU memory not resident?); check MemAvailable / dmesg, retry later\n"); exit(3); }
        }
    }
    fprintf(stderr, "self-check ok\n");
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(ls, (sockaddr *) &a, sizeof a) || listen(ls, 4)) { perror("bind"); return 1; }
    fprintf(stderr, "serving on %d (busy-poll)\n", port);
    if (getenv("K3X_MEMWARM")) memwarm_start(atof(getenv("K3X_MEMWARM")));
    if (const char * rt = getenv("K3X_RT")) if (atoi(rt) > 0) {   // SCHED_FIFO request thread; then block after 1 ms idle
        sched_param sp{}; sp.sched_priority = atoi(rt);
        if (sched_setscheduler(0, SCHED_FIFO | SCHED_RESET_ON_FORK, &sp)) perror("K3X_RT sched_setscheduler");
        else fprintf(stderr, "K3X_RT: request thread SCHED_FIFO %d\n", sp.sched_priority);
        g_rx_spin_ns = 1000 * 1000L;
    }
    if (const char * rs = getenv("K3X_RX_SPIN_US")) g_rx_spin_ns = atol(rs) < 0 ? -1 : atol(rs) * 1000L;
    if (g_rx_spin_ns >= 0) fprintf(stderr, "K3X_RX_SPIN_US: %ld (then poll)\n", g_rx_spin_ns / 1000);
    long zero_rows = 0; int nconn = 0;
    for (;;) {
        const int c = accept(ls, nullptr, nullptr);
        if (c < 0) { if (errno == EINTR) continue; perror("accept"); return 1; }
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        int fw = -1;   // K3X_FORCEWAKE=<i915_forcewake_user>: held open while a client is connected (GT out of RC6)
        if (const char * fp = getenv("K3X_FORCEWAKE")) { fw = open(fp, O_RDONLY); if (fw < 0) perror(fp); }
        nconn++;
        for (;;) {
            g_warm_on.store(1, std::memory_order_relaxed);
            const bool got = rd_spin(c, hdr, 17 * 4);
            g_warm_on.store(0, std::memory_order_relaxed);
            if (!got || !rd_spin(c, x, (size_t) D * 4)) break;
            const int k = hdr[0];
            if (k < 1 || k > 16) break;
            int ki = -1; bool ok = true;
            for (int i = 0; ok && i < k; i++) {
                auto it = map.find(hdr[1 + i]);
                if (it == map.end() || (i && it->second.kind != ki)) ok = false;
                else { ki = it->second.kind; set_w(i, kinds[ki], it->second.p); }
            }
            if (!ok) { if (!wr_all(c, nanrows.data(), (size_t) k * D * 4)) break; continue; }
            const Kind & K = kinds[ki];
            run(K, k);
            bool okw = true;
            for (int j = 0; j < k; j++) {
                if (stream || j == 0) done(stream ? j : k - 1);
                const float * r = rows + (size_t) j * D; bool nz = false;
                for (int i = 0; i < D && !nz; i++) nz = r[i] != 0.0f;
                if (!nz) { zero_rows++; r = nanrows.data(); }
                if (okw) okw = wr_all(c, r, (size_t) D * 4);   // on a send error still wait for the kernels (rows, W reused)
            }
            if (!okw) break;
        }
        close(c);
        if (fw >= 0) close(fw);
        if (zero_rows) fprintf(stderr, "conn %d: %ld all-zero rows sent as NaN so far\n", nconn, zero_rows);
    }
}

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: see source\n"); return 1; }
    if (!strcmp(argv[5], "serveidx")) return serve_index(argv[1], argv[2], argv[3], atoi(argv[6]));
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
    const bool hostw = getenv("G53X_HOSTW") && atoi(getenv("G53X_HOSTW"));   // weights in host USM (serveidx's arena)
    auto walloc = [&](size_t n) { return hostw ? sycl::malloc_host<uint8_t>(n, q) : sycl::malloc_device<uint8_t>(n, q); };
    uint8_t * wg = walloc(gsz * slots), * wu = walloc(usz * slots), * wd = walloc(dsz * slots);
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
    B.x = sycl::malloc_host<float>(D, q); B.W = sycl::malloc_host<const uint8_t *>(3 * 16, q); B.out = sycl::malloc_host<float>(NE * D, q);
    B.xq = sycl::malloc_device<int8_t>(D, q); B.xd = sycl::malloc_device<float>(D / 32, q); B.xs = sycl::malloc_device<float>(D / 32, q);
    B.h = sycl::malloc_device<float>(NE * F, q); B.hq = sycl::malloc_device<int8_t>(NE * F, q);
    B.hd = sycl::malloc_device<float>(NE * F / 32, q); B.hs = sycl::malloc_device<float>(NE * F / 32, q);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < D; i++) B.x[i] = nd(rng);   // same draw as glmbench
    if (getenv("G53X_XIN")) { auto xb = slurp(getenv("G53X_XIN"), D * 4); memcpy(B.x, xb.data(), D * 4); }

    GateUpFn gu = pick_gu(tg, tu, D, F); DownFn dn = pick_dn(td, D, F);
    if (!gu) { fprintf(stderr, "gate/up %d/%d not instantiated\n", tg, tu); exit(1); }
    if (!dn) { fprintf(stderr, "down %d not instantiated\n", td); exit(1); }
    auto set_ids = [&](const int * ids, int k) {   // slot ids -> the kernels' weight pointer table
        for (int i = 0; i < k; i++) { B.W[3 * i] = wg + (size_t) ids[i] * gsz; B.W[3 * i + 1] = wu + (size_t) ids[i] * usz; B.W[3 * i + 2] = wd + (size_t) ids[i] * dsz; }
    };
    std::vector<sycl::event> ev(4);
    const bool fuseq = !getenv("G53X_FUSEQ") || atoi(getenv("G53X_FUSEQ"));   // check mode runs both and compares
    auto run = [&](int k) {
        if (fuseq) {   // two launches: q8_1 inside the matvec kernels
            ev[1] = gu(q, B.W, k, B.x, nullptr, nullptr, nullptr, B.h, B.gu);
            ev[3] = dn(q, B.W, k, B.h, nullptr, nullptr, nullptr, B.out);
        } else {
            quantize(q, B.x, B.xq, B.xd, B.xs, D / 32);
            ev[1] = gu(q, B.W, k, nullptr, B.xq, B.xd, B.xs, B.h, B.gu);
            quantize(q, B.h, B.hq, B.hd, B.hs, k * F / 32);
            ev[3] = dn(q, B.W, k, nullptr, B.hq, B.hd, B.hs, B.out);
        }
        ev[3].wait();
    };

    if (check) {
        { int ids[NE]; for (int i = 0; i < NE; i++) ids[i] = i; set_ids(ids, NE); }
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
            set_ids(perm.data(), k);
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
