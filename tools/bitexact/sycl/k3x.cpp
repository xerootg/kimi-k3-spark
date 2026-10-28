// k3x -- standalone bit-exact SYCL routed-expert FFN for Kimi K3 (one decode token), emulating the Spark's ggml-cuda kernels.
// usage:
//   k3x RECORDS N_RECORDS GATE_TYPE DOWN_TYPE check [N]            device vs host reference on random ids/x
//   k3x RECORDS N_RECORDS GATE_TYPE DOWN_TYPE bench k1,k2,.. ITERS  timing (p50/p90, GB/s)
//   k3x RECORDS N_RECORDS GATE_TYPE DOWN_TYPE serve PORT            rrclient/moebench wire protocol, busy-polling
//   RECORDS = "synthetic" makes random IQ records in memory.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>
#include <cerrno>
#include <cmath>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "k3x_tables.h"
#include "k3x_math.h"
#include "k3x_act.h"


#include <sycl/ext/oneapi/dot_product.hpp>
static inline float h2f_dev(uint16_t h) {
#ifdef __SYCL_DEVICE_ONLY__
    return (float) sycl::bit_cast<sycl::half>(h);
#else
    return k3_h2f(h);
#endif
}
static inline int dp4a_dev(int a, int b, int c) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::ext::oneapi::dot_acc(a, b, c);
#else
    return k3_dp4a(a, b, c);
#endif
}
// Device tails without FTZ emulation. Proof that no denormal can occur in the matvec stage (so .FTZ is a no-op there):
// every operand is h2f(half) (>= 2^-24 or 0), a small int, -0.875/-1.125, or a product/sum of those; all exact
// values are multiples of 2^-24*2^-24*2^-3 and every rounded value stays >= 2^-75 in magnitude or is 0, far above
// FLT_MIN = 2^-126, and sums of such values (row reduction) are multiples of 2^-99*... likewise never denormal.
// Validated against the Spark dumps (validate mode).
static inline float iq1s_tail(float dhf, int qh, int sumi, float dsx, float dsy) {
    const float dW = dhf * (float) (((qh >> 11) & 0x0E) + 1);
    const float delta = (qh & 0x8000) ? -1.125f : -0.875f;
    const float inner = K3_FMA(dsx, (float) sumi, dsy * delta);
    return K3_FMA(dW, inner, 0.0f);
}
static inline float iq2xxs_tail(float dhf, int sumi, float q8df) { return K3_FMA(dhf * q8df, (float) sumi, 0.0f); }
// device fast paths: one 256-block, all 8 CUDA lanes (kqs 0..7) -> out[8]; integer part reordered freely (exact),
// float tails are the shared k3_*_finish functions.
static inline void blk_iq1_s(const uint8_t * blk, const k3_q8_1 * y8, const sycl::float2 * dsf, const uint32_t * grid, float * out) {
    const uint16_t * w = (const uint16_t *) blk;
    const uint16_t dh = w[0];
#pragma unroll
    for (int l = 0; l < 8; l++) {
        const uint32_t qsp = (uint32_t) w[1 + 2 * l] | ((uint32_t) w[2 + 2 * l] << 16);
        const int qh = w[17 + l];
        const int * u = (const int *) y8[l].qs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int g = (int) grid[((qsp >> (8 * j)) & 0xff) | (((qh >> (3 * j)) & 7) << 8)];
            sumi = dp4a_dev(g & 0x0F0F0F0F, u[2 * j], sumi);
            sumi = dp4a_dev((g >> 4) & 0x0F0F0F0F, u[2 * j + 1], sumi);
        }
        out[l] = iq1s_tail(h2f_dev(dh), qh, sumi, dsf[l].x(), dsf[l].y());
    }
}
static inline void blk_iq2_xxs(const uint8_t * blk, const k3_q8_1 * y8, const sycl::float2 * dsf, const uint64_t * grid, float * out) {
    const uint16_t * w = (const uint16_t *) blk;
    const uint16_t dh = w[0];
#pragma unroll
    for (int l = 0; l < 8; l++) {
        const uint32_t q2 = (uint32_t) w[1 + 4 * l] | ((uint32_t) w[2 + 4 * l] << 16);
        const uint32_t aux32 = (uint32_t) w[3 + 4 * l] | ((uint32_t) w[4 + 4 * l] << 16);
        const int * u = (const int *) y8[l].qs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint64_t gp = grid[(q2 >> (8 * j)) & 0xff];
            const uint32_t v = (aux32 >> (7 * j)) & 0xff;
            const uint32_t sgn = (v ^ ((sycl::popcount(v) & 1) << 7)) * 0x01010101u;
            const uint32_t m0 = (((sgn & 0x08040201u) + 0x7f7f7f7fu) & 0x80808080u) >> 7;
            const uint32_t m1 = (((sgn & 0x80402010u) + 0x7f7f7f7fu) & 0x80808080u) >> 7;
            const uint32_t g0 = ((uint32_t) gp ^ (m0 * 0xffu)) + m0;
            const uint32_t g1 = ((uint32_t) (gp >> 32) ^ (m1 * 0xffu)) + m1;
            sumi = dp4a_dev((int) g0, u[2 * j], sumi);
            sumi = dp4a_dev((int) g1, u[2 * j + 1], sumi);
        }
        const int ls = (int) (aux32 >> 27 | 1);
        out[l] = iq2xxs_tail(h2f_dev(dh), sumi * ls / 8, dsf[l].x());
    }
}
static inline void blk_any(int t, const uint8_t * blk, const k3_q8_1 * y8, const sycl::float2 * dsf, const uint32_t * g1, const uint64_t * g2, float * out) {
    if (t == T_IQ1_S) blk_iq1_s(blk, y8, dsf, g1, out); else blk_iq2_xxs(blk, y8, dsf, g2, out);
}

static constexpr int D = K3_D, F = K3_F;
#ifndef K3X_SLM_REDUCE
#define K3X_SLM_REDUCE 0
#endif
#ifndef EXP
#define EXP 0
#endif
#ifndef RPW
#define RPW 16   // rows per work-group
#endif

static sycl::queue * g_queue = nullptr;   // one queue/context shared by every Engine (USM tables are shared)
static void async_err(sycl::exception_list el) { for (auto & e : el) { try { std::rethrow_exception(e); } catch (sycl::exception & x) { fprintf(stderr, "SYCL async error: %s\n", x.what()); } } }
struct RecPtrs { const uint8_t * p[16]; };
struct Engine {
    sycl::queue q;
    int tg, td, E;
    size_t rbg, rbd, gsz, dsz, rec;   // row bytes, matrix bytes, record bytes
    uint8_t * W = nullptr;
    uint32_t * g1 = nullptr; uint64_t * g2 = nullptr;
    float * x = nullptr; int * ids = nullptr; float * rows = nullptr; const uint8_t ** recs = nullptr; bool ext_recs = false; bool spin = false;
    float * xd = nullptr, * rowsd = nullptr;   // device copies: explicit memcpy in/out (host USM read/written by kernels was not reliably coherent)   // host USM (zero copy on the iGPU)
    k3_q8_1 * yq = nullptr, * yq2 = nullptr; sycl::float2 * dsf = nullptr, * dsf2 = nullptr;
    float * gbuf = nullptr, * ubuf = nullptr, * hbuf = nullptr;
    k3_mufu_tabs mt{};
    int nwg = getenv("K3X_NWG") ? atoi(getenv("K3X_NWG")) : 56;
    bool prof = false; std::vector<sycl::event> ev; double kt[4] = {0, 0, 0, 0}; long nrun = 0;

    static sycl::property_list props(bool p) {
        if (p) return sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
        return sycl::property_list{sycl::property::queue::in_order()};
    }
    static sycl::queue & shared_queue() {
        if (!g_queue) g_queue = new sycl::queue(sycl::gpu_selector_v, async_err, props(getenv("K3X_PROF") != nullptr));
        return *g_queue;
    }
    Engine(int tg_, int td_, int E_) : q(shared_queue()), tg(tg_), td(td_), E(E_) {
        prof = getenv("K3X_PROF") != nullptr;
        rbg = (size_t) D / K3_QK * k3_blk_bytes(tg); rbd = (size_t) F / K3_QK * k3_blk_bytes(td);
        gsz = rbg * F; dsz = rbd * D; rec = 2 * gsz + dsz;
        if (E > 0) W = sycl::malloc_device<uint8_t>(rec * E, q);
        g1 = sycl::malloc_device<uint32_t>(2048, q); g2 = sycl::malloc_device<uint64_t>(256, q);
        q.memcpy(g1, iq1s_grid_gpu, 2048 * 4); q.memcpy(g2, iq2xxs_grid, 256 * 8);
        x = sycl::malloc_host<float>(D, q); ids = sycl::malloc_host<int>(16, q); recs = sycl::malloc_host<const uint8_t *>(16, q); rows = sycl::malloc_host<float>(16 * D, q);
        xd = sycl::malloc_device<float>(D, q); rowsd = sycl::malloc_device<float>(16 * D, q);
        yq = sycl::malloc_device<k3_q8_1>(D / 32, q); yq2 = sycl::malloc_device<k3_q8_1>(16 * F / 32, q);
        dsf = sycl::malloc_device<sycl::float2>(D / 32, q); dsf2 = sycl::malloc_device<sycl::float2>(16 * F / 32, q);
        gbuf = sycl::malloc_device<float>(16 * F, q); ubuf = sycl::malloc_device<float>(16 * F, q); hbuf = sycl::malloc_device<float>(16 * F, q);
        q.wait();
        fprintf(stderr, "device: %s; record %zu B (gate/up %zu, down %zu), %d records = %.1f MB\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), rec, gsz, dsz, E, rec * E / 1e6);
    }

    void upload(const uint8_t * host, size_t e0, size_t n) { q.memcpy(W + e0 * rec, host, n * rec).wait(); }

    // nmat matrices per expert (gate/up: 2, down: 1) of nrows rows x nblk q-blocks of type t, row bytes rb,
    // first matrix at moff in the record, second at moff + mstride; y/dsf advance by ystride blocks per expert slot.
    sycl::event matvec(int k, int nmat, int nrows, int nblk, int t, size_t rb, size_t moff, size_t mstride, int /*unused*/,
                       const k3_q8_1 * y, const sycl::float2 * dsfp, int ystride, float * out0, float * out1) {
        const uint8_t * W_ = W; const uint32_t * g1_ = g1; const uint64_t * g2_ = g2; RecPtrs recs_; for (int i = 0; i < 16; i++) recs_.p[i] = recs[i];
        const int n16 = (int) (RPW * rb / 16), gpm = nrows / RPW;
        const int total = k * nmat * gpm;
        const int ngroups = std::min(total, nwg);
        const int gn = t == T_IQ1_S ? 2048 : 512;   // grid table words staged in SLM
        return q.submit([&](sycl::handler & h) {
            sycl::local_accessor<sycl::uint4, 1> T(sycl::range<1>(n16), h);
            sycl::local_accessor<float, 1> P(sycl::range<1>(K3X_SLM_REDUCE ? RPW * 128 : 1), h);
            sycl::local_accessor<uint32_t, 1> G(sycl::range<1>(gn), h);
            h.parallel_for(sycl::nd_range<1>((size_t) ngroups * RPW * 16, RPW * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid = it.get_local_id(0);
                const uint32_t * gsrc = t == T_IQ1_S ? g1_ : (const uint32_t *) g2_;
                for (int i = lid; i < gn; i += RPW * 16) G[i] = gsrc[i];
                const uint32_t * gl1 = G.get_multi_ptr<sycl::access::decorated::no>().get();
                const uint64_t * gl2 = (const uint64_t *) gl1;
                const int r = lid / 16, b = lid % 16;
                // software pipeline: the next task's tile is loaded into registers while this one computes
                constexpr int NR = 4;   // >= ceil(n16 / (RPW*16)) for RPW 16 (IQ2_XXS gate: 924 uint4)
                sycl::uint4 nx[NR];
                auto src_of = [&](int task) {
                    const int rg = task % gpm, m = (task / gpm) % nmat, e = task / (gpm * nmat);
                    return (const sycl::uint4 *) (recs_.p[e] + moff + m * mstride + rg * RPW * rb);
                };
                int task = it.get_group(0);
                if (task < total) { const sycl::uint4 * src = src_of(task);
#pragma unroll
                    for (int j = 0; j < NR; j++) { const int i = lid + j * RPW * 16; if (i < n16) nx[j] = src[i]; } }
                for (; task < total; task += ngroups) {
                    const int rg = task % gpm, m = (task / gpm) % nmat, e = task / (gpm * nmat);
                    const int row0 = rg * RPW;
                    sycl::group_barrier(it.get_group());   // previous task done with T/P (and G loaded)
#pragma unroll
                    for (int j = 0; j < NR; j++) { const int i = lid + j * RPW * 16; if (i < n16) T[i] = nx[j]; }
                    sycl::group_barrier(it.get_group());
                    if (task + ngroups < total) { const sycl::uint4 * src = src_of(task + ngroups);
#pragma unroll
                        for (int j = 0; j < NR; j++) { const int i = lid + j * RPW * 16; if (i < n16) nx[j] = src[i]; } }
                    float o[8];
                    if (b < nblk) {
#if EXP == 1
                        for (int l = 0; l < 8; l++) o[l] = (float) ((const uint8_t *) &T[0])[r * rb + b * k3_blk_bytes(t) + l];
#else
                        blk_any(t, (const uint8_t *) &T[0] + r * rb + b * k3_blk_bytes(t), y + e * ystride + b * 8, dsfp + e * ystride + b * 8, gl1, gl2, o);
#endif
                    } else {
                        for (int l = 0; l < 8; l++) o[l] = 0.0f;
                    }
#if K3X_SLM_REDUCE
                    float * pp = &P[r * 128 + b * 8];
                    for (int l = 0; l < 8; l++) pp[l] = o[l];
                    sycl::group_barrier(it.get_group());
                    if (lid < RPW) (m ? out1 : out0)[e * nrows + row0 + lid] = k3_row_reduce(&P[lid * 128]);
#else
                    // CUDA thread tid = 8*b + l of this row lives in lane b, register l. Same adds, same order as
                    // k3_row_reduce: s_x = ((p_x + p_{x+32}) + p_{x+64}) + p_{x+96}, then the xor-butterfly tree.
                    auto sg = it.get_sub_group();
                    float v[8];
#pragma unroll
                    for (int l = 0; l < 8; l++) {
                        const float a1 = sycl::select_from_group(sg, o[l], (b + 4) & 15);
                        const float a2 = sycl::select_from_group(sg, o[l], (b + 8) & 15);
                        const float a3 = sycl::select_from_group(sg, o[l], (b + 12) & 15);
                        v[l] = ((o[l] + a1) + a2) + a3;           // valid in lanes 0..3 (x = 8b + l < 32)
                    }
#pragma unroll
                    for (int l = 0; l < 8; l++) v[l] = v[l] + sycl::select_from_group(sg, v[l], (b + 2) & 15);  // off 16: lanes 0,1
#pragma unroll
                    for (int l = 0; l < 8; l++) v[l] = v[l] + sycl::select_from_group(sg, v[l], (b + 1) & 15);  // off 8: lane 0
#pragma unroll
                    for (int l = 0; l < 4; l++) v[l] = v[l] + v[l + 4];   // off 4
                    v[0] = v[0] + v[2]; v[1] = v[1] + v[3];                // off 2
                    v[0] = v[0] + v[1];                                     // off 1
                    if (b == 0) (m ? out1 : out0)[e * nrows + row0 + r] = v[0];
#endif
                }
            });
        });
    }

    // down stage only, from h already in hbuf (validation of quant(h) + down independent of the MUFU chain)
    void run_from_h(int k) {
        k3_q8_1 * yq2_ = yq2; sycl::float2 * dsf2_ = dsf2; const float * hb = hbuf; const k3_mufu_tabs mt_ = mt;
        if (!ext_recs) for (int i = 0; i < k; i++) recs[i] = W + (size_t) ids[i] * rec;
        q.parallel_for(sycl::range<1>((size_t) k * F / 32), [=](sycl::id<1> i) {
            k3_quantize_block(hb + i[0] * 32, yq2_ + i[0], mt_);
            dsf2_[i[0]] = sycl::float2(k3_h2f(yq2_[i[0]].d), k3_h2f(yq2_[i[0]].s));
        });
        matvec(k, 1, D, F / K3_QK, td, rbd, 2 * gsz, 0, 0, yq2, dsf2, F / 32, rowsd, nullptr);
        q.memcpy(rows, rowsd, (size_t) k * D * 4);
        q.wait_and_throw();
    }

    // one token: ids[0..k), x[D] -> rows[k][D]
    void run(int k) {
        const int tg_ = tg, td_ = td; const size_t rbg_ = rbg, rbd_ = rbd, gsz_ = gsz, rec_ = rec;
        const uint8_t * W_ = W; const uint32_t * g1_ = g1; const uint64_t * g2_ = g2;
        q.memcpy(xd, x, D * 4);
        const float * x_ = xd; float * rows_ = rowsd;
        if (!ext_recs) for (int i = 0; i < k; i++) recs[i] = W + (size_t) ids[i] * rec;
        k3_q8_1 * yq_ = yq, * yq2_ = yq2; sycl::float2 * dsf_ = dsf, * dsf2_ = dsf2;
        float * gb = gbuf, * ub = ubuf, * hb = hbuf; const k3_mufu_tabs mt_ = mt;
        ev.clear();
        ev.push_back(q.parallel_for(sycl::range<1>(D / 32), [=](sycl::id<1> i) {
            k3_quantize_block(x_ + i[0] * 32, yq_ + i[0], mt_);
            dsf_[i[0]] = sycl::float2(k3_h2f(yq_[i[0]].d), k3_h2f(yq_[i[0]].s));
        }));
        // gate and up: separate work-groups (m = 0 gate, 1 up), each RPW contiguous rows staged in SLM
        ev.push_back(matvec(k, 2, F, D / K3_QK, tg_, rbg_, 0, gsz_, F / 32 * 0, yq_, dsf_, 0, gb, ub));
        // SiTU, one work-item per element (the MUFU table lookups are latency bound: maximize parallelism)
        q.parallel_for(sycl::range<1>((size_t) k * F), [=](sycl::id<1> i) { hb[i[0]] = k3_act(gb[i[0]], ub[i[0]], mt_); });
        ev.push_back(q.parallel_for(sycl::range<1>((size_t) k * F / 32), [=](sycl::id<1> i) {
            k3_quantize_block(hb + i[0] * 32, yq2_ + i[0], mt_);
            dsf2_[i[0]] = sycl::float2(k3_h2f(yq2_[i[0]].d), k3_h2f(yq2_[i[0]].s));
        }));
        ev.push_back(matvec(k, 1, D, F / K3_QK, td_, rbd_, 2 * gsz_, 0, 0, yq2_, dsf2_, F / 32, rows_, nullptr));
        ev.push_back(q.memcpy(rows, rowsd, (size_t) k * D * 4));
        if (spin) { auto & last = ev.back(); while (last.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {} }
        q.wait_and_throw();
        if (prof) {
            for (int i = 0; i < 4; i++)
                kt[i] += (ev[i].get_profiling_info<sycl::info::event_profiling::command_end>() -
                          ev[i].get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
            nrun++;
        }
    }
};

// host reference, same functions
static void host_ref(const uint8_t * W, size_t rec, int tg, int td, const int * ids, int k, const float * x, float * out,
                     const k3_mufu_tabs & mt, float * gdump = nullptr, float * udump = nullptr) {
    const size_t rbg = (size_t) D / K3_QK * k3_blk_bytes(tg), rbd = (size_t) F / K3_QK * k3_blk_bytes(td), gsz = rbg * F;
    std::vector<k3_q8_1> yq(D / 32), yq2(F / 32);
    for (int i = 0; i < D / 32; i++) k3_quantize_block(x + i * 32, &yq[i], mt);
    std::vector<float> g(F), u(F), a(F), P(128);
    for (int e = 0; e < k; e++) {
        const uint8_t * base = W + (size_t) ids[e] * rec;
        for (int m = 0; m < 2; m++)
            for (int row = 0; row < F; row++) {
                for (int t = 0; t < 128; t++) {
                    const int b = t / 8, l = t % 8;
                    P[t] = b < D / K3_QK ? k3_vdot(tg, base + m * gsz + row * rbg + b * k3_blk_bytes(tg), &yq[b * 8], l, iq1s_grid_gpu, iq2xxs_grid) : 0.0f;
                }
                (m ? u : g)[row] = k3_row_reduce(P.data());
            }
        if (gdump) memcpy(gdump + e * F, g.data(), F * 4);
        if (udump) memcpy(udump + e * F, u.data(), F * 4);
        for (int i = 0; i < F; i++) a[i] = k3_act(g[i], u[i], mt);
        for (int i = 0; i < F / 32; i++) k3_quantize_block(&a[i * 32], &yq2[i], mt);
        for (int row = 0; row < D; row++) {
            for (int t = 0; t < 128; t++) {
                const int b = t / 8, l = t % 8;
                P[t] = b < F / K3_QK ? k3_vdot(td, base + 2 * gsz + row * rbd + b * k3_blk_bytes(td), &yq2[b * 8], l, iq1s_grid_gpu, iq2xxs_grid) : 0.0f;
            }
            out[e * D + row] = k3_row_reduce(P.data());
        }
    }
}

static void synth_records(std::vector<uint8_t> & buf, int tg, int td, int E, size_t rec, size_t gsz, uint32_t seed) {
    std::mt19937 rng(seed);
    buf.resize(rec * E);
    for (auto & c : buf) c = rng() & 0xff;
    auto fix = [&](uint8_t * m, int t, size_t nblocks) {   // sane block scales (half in [2^-10, 2^-7])
        const int bb = k3_blk_bytes(t);
        for (size_t i = 0; i < nblocks; i++) { uint16_t h = k3_f2h(std::ldexp(1.0f + (rng() % 1000) / 1000.0f, -10 + (int) (rng() % 3))); memcpy(m + i * bb, &h, 2); }
    };
    for (int e = 0; e < E; e++) {
        uint8_t * r = buf.data() + e * rec;
        fix(r, tg, gsz / k3_blk_bytes(tg)); fix(r + gsz, tg, gsz / k3_blk_bytes(tg)); fix(r + 2 * gsz, td, (rec - 2 * gsz) / k3_blk_bytes(td));
    }
}


static std::vector<char> slurp(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb"); if (!f) { perror(p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(n); if (fread(v.data(), 1, n, f) != (size_t) n) exit(1); fclose(f); return v;
}
static k3_mufu_tabs host_mt{};
static std::vector<char> tab_host[4];
// loads ref/mufu/{rcp,ex2p,ex2n,tanh}.bin (K3X_MUFU=dir) into host memory once and into this engine's device memory
static k3_mufu_tabs dev_mt_cache{}; static bool dev_mt_loaded = false;
static void load_mufu(Engine & en) {
    if (dev_mt_loaded) { en.mt = dev_mt_cache; return; }
    const char * dir = getenv("K3X_MUFU") ? getenv("K3X_MUFU") : "../ref/mufu";
    const char * names[4] = {"rcp", "ex2p", "ex2n", "tanh"};
    const uint32_t * dev[4] = {nullptr, nullptr, nullptr, nullptr}; uint32_t n[4] = {0, 0, 0, 0};
    for (int t = 0; t < 4; t++) {
        std::string p = std::string(dir) + "/" + names[t] + ".bin";
        FILE * f = fopen(p.c_str(), "rb");
        if (!f) { fprintf(stderr, "WARNING: no %s, IEEE placeholder\n", p.c_str()); continue; }
        fclose(f);
        if (tab_host[t].empty()) tab_host[t] = slurp(p);
        n[t] = (uint32_t) (tab_host[t].size() / 4);
        uint32_t * d = sycl::malloc_device<uint32_t>(n[t], en.q);
        en.q.memcpy(d, tab_host[t].data(), (size_t) n[t] * 4).wait();
        dev[t] = d;
        fprintf(stderr, "MUFU %s table: %u entries\n", names[t], n[t]);
    }
    en.mt = k3_mufu_tabs{dev[0], dev[1], dev[2], dev[3], n[1], n[2], n[3]};
    dev_mt_cache = en.mt; dev_mt_loaded = true;
    host_mt = k3_mufu_tabs{(const uint32_t *) tab_host[0].data(), (const uint32_t *) tab_host[1].data(), (const uint32_t *) tab_host[2].data(),
                           (const uint32_t *) tab_host[3].data(), n[1], n[2], n[3]};
    for (int t = 0; t < 4; t++) if (!n[t]) { if (t == 0) host_mt.rcp = nullptr; if (t == 1) host_mt.ex2p = nullptr; if (t == 2) host_mt.ex2n = nullptr; if (t == 3) host_mt.tanh = nullptr; }
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }


static bool rd_spin(int fd, void * p, size_t n) {
    char * c = (char *) p;
    while (n) { ssize_t r = recv(fd, c, n, MSG_DONTWAIT); if (r == 0) return false; if (r < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK) continue; return false; } c += r; n -= r; }
    return true;
}
static bool wr_all(int fd, const void * p, size_t n) { const char * c = (const char *) p; while (n) { ssize_t r = send(fd, c, n, 0); if (r <= 0) return false; c += r; n -= r; } return true; }

// serve agent C's remote set: IDX lines "<global_id> <tg> <td> <byte_offset>", BIN = records back to back.
// Weights go to device USM in chunks of <= 1 GiB (no >4 GiB buffers needed); one Engine (scratch + kernels) per (tg, td).
static int serve_index(const char * idxp, const char * binp, int port) {
    struct R { long gid; int tg, td; size_t off; };
    std::vector<R> idx; { FILE * f = fopen(idxp, "r"); if (!f) { perror(idxp); return 1; } R r; while (fscanf(f, "%ld %d %d %zu", &r.gid, &r.tg, &r.td, &r.off) == 4) idx.push_back(r); fclose(f); }
    std::vector<Engine *> eng; auto get = [&](int tg, int td) -> Engine * { for (auto * e : eng) if (e->tg == tg && e->td == td) return e; eng.push_back(new Engine(tg, td, 0)); eng.back()->spin = true; load_mufu(*eng.back()); return eng.back(); };
    std::unordered_map<long, std::pair<Engine *, const uint8_t *>> map;
    FILE * f = fopen(binp, "rb"); if (!f) { perror(binp); return 1; }
    const size_t CH = (size_t) 1 << 30; uint8_t * chunk = nullptr; size_t used = CH; std::vector<uint8_t> buf;
    double t0 = now_ms(); size_t tot = 0;
    for (auto & r : idx) {
        Engine * e = get(r.tg, r.td);
        if (used + e->rec > CH) { chunk = sycl::malloc_device<uint8_t>(CH, e->q); if (!chunk) { fprintf(stderr, "device alloc failed after %.1f GB\n", tot / 1e9); return 1; } used = 0; }
        buf.resize(e->rec);
        if (fseeko(f, (off_t) r.off, SEEK_SET) || fread(buf.data(), 1, e->rec, f) != e->rec) { fprintf(stderr, "short read gid %ld\n", r.gid); return 1; }
        e->q.memcpy(chunk + used, buf.data(), e->rec).wait();
        map[r.gid] = {e, chunk + used}; used += e->rec; tot += e->rec;
    }
    fclose(f);
    for (auto & v : tab_host) std::vector<char>().swap(v);   // host copies only serve the host reference
    fprintf(stderr, "loaded %zu records (%.1f GB) in %.1f s, %zu engine kinds\n", idx.size(), tot / 1e9, (now_ms() - t0) / 1e3, eng.size());
    for (auto * e : eng) { e->ext_recs = true; for (int i = 0; i < D; i++) e->x[i] = 0.1f; for (auto & kv : map) if (kv.second.first == e) { e->recs[0] = kv.second.second; break; } e->run(1); }
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(ls, (sockaddr *) &a, sizeof a) || listen(ls, 4)) { perror("bind"); return 1; }
    fprintf(stderr, "serving on %d (busy-poll)\n", port);
    std::vector<float> nanrows(16 * D, NAN), xin(D);
    for (;;) {
        int c = accept(ls, nullptr, nullptr);
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        int32_t hdr[17];
        while (rd_spin(c, hdr, 17 * 4) && rd_spin(c, xin.data(), D * 4)) {
            const int k = hdr[0];
            if (k < 1 || k > 16) break;
            auto it0 = map.find(hdr[1]);
            Engine * e = it0 == map.end() ? nullptr : it0->second.first;
            bool ok = e != nullptr;
            for (int i = 0; ok && i < k; i++) { auto it = map.find(hdr[1 + i]); if (it == map.end() || it->second.first != e) ok = false; else e->recs[i] = it->second.second; }
            if (!ok) { if (!wr_all(c, nanrows.data(), (size_t) k * D * 4)) break; continue; }
            memcpy(e->x, xin.data(), D * 4);
            e->run(k);
            if (!wr_all(c, e->rows, (size_t) k * D * 4)) break;
        }
        close(c);
    }
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: k3x RECORDS N TG TD check|bench|serve ...\n"); return 1; }
    std::string path = argv[1]; int E = atoi(argv[2]), tg = atoi(argv[3]), td = atoi(argv[4]); std::string mode = argv[5];
    if (mode == "serveidx") return serve_index(argv[1], argv[2], atoi(argv[6]));
    Engine en(tg, td, E);
    std::vector<uint8_t> host;
    if (getenv("K3X_SPIN")) en.spin = true;
    load_mufu(en);
    if (path == "synthetic") synth_records(host, tg, td, E, en.rec, en.gsz, 42);
    else {
        FILE * f = fopen(path.c_str(), "rb"); if (!f) { perror(path.c_str()); return 1; }
        host.resize(en.rec * E);
        if (fread(host.data(), 1, host.size(), f) != host.size()) { fprintf(stderr, "short read\n"); return 1; }
        fclose(f);
    }
    en.upload(host.data(), 0, E);
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.f, 1.f);

    if (mode == "check") {
        int n = argc > 6 ? atoi(argv[6]) : 4;
        long tot = 0, same = 0; double maxrel = 0;
        for (int it = 0; it < n; it++) {
            const int k = 1 + it % 4;
            for (int i = 0; i < D; i++) en.x[i] = nd(rng) * 0.5f;
            for (int i = 0; i < k; i++) en.ids[i] = rng() % E;
            en.run(k);
            std::vector<float> ref(k * D), hg(k * F), hu(k * F), dg(k * F), du(k * F);
            host_ref(host.data(), en.rec, tg, td, en.ids, k, en.x, ref.data(), host_mt, hg.data(), hu.data());
            en.q.memcpy(dg.data(), en.gbuf, k * F * 4); en.q.memcpy(du.data(), en.ubuf, k * F * 4).wait();
            for (int e = 0; e < k; e++) {
                int bg = 0, bu = 0, bo = 0;
                for (int i = 0; i < F; i++) { bg += k3_f2u(hg[e * F + i]) != k3_f2u(dg[e * F + i]); bu += k3_f2u(hu[e * F + i]) != k3_f2u(du[e * F + i]); }
                for (int i = 0; i < D; i++) bo += k3_f2u(ref[e * D + i]) != k3_f2u(en.rows[e * D + i]);
                if (bg || bu || bo) printf("  it %d k %d slot %d id %d: gate diff %d up diff %d out diff %d\n", it, k, e, en.ids[e], bg, bu, bo);
            }
            for (int i = 0; i < k * D; i++) {
                tot++; if (k3_f2u(ref[i]) == k3_f2u(en.rows[i])) same++;
                else if (getenv("K3X_SHOW") && tot - same < 6) printf("  e%d row %d: ref %.9g dev %.9g\n", i / D, i % D, ref[i], en.rows[i]);
                else { double r = fabs((ref[i] - en.rows[i]) / (fabs(ref[i]) + 1e-30)); if (r > maxrel) maxrel = r; }
            }
        }
        printf("check: %ld / %ld rows bit-identical device vs host (max rel diff otherwise %.3g)\n", same, tot, maxrel);
        return same == tot ? 0 : 2;
    }
    if (mode == "hdebug") {   // host: h from dumped g,u with the MUFU emulation vs dumped h
        std::string dir = argv[6];
        auto G = slurp(dir + "/g.bin"), U = slurp(dir + "/u.bin"), H = slurp(dir + "/h.bin");
        const float * g = (const float *) G.data(), * u = (const float *) U.data(), * h = (const float *) H.data();
        size_t n = G.size() / 4, bad = 0, shown = 0;
        for (size_t i = 0; i < n; i++) {
            float m = k3_act(g[i], u[i], host_mt);
            if (k3_f2u(m) != k3_f2u(h[i])) { bad++; if (shown++ < 12) {
                float a = fmul_z(nv_tanh(fmul_z(g[i], 0.25f), host_mt), 4.0f), e = nv_ex2(fmul_z(g[i], k3_u2f(0xBFB8AA3Bu)), host_mt);
                float c = fmul_z(nv_tanh(fmul_z(u[i], k3_u2f(0x3D23D70Au)), host_mt), 25.0f);
                printf("  i %zu g %.9g u %.9g h_ref %.9g (%08x) mine %.9g (%08x) a %.9g e %.9g c %.9g ieee %.9g\n", i, g[i], u[i], h[i], k3_f2u(h[i]), m, k3_f2u(m), a, e, c,
                       4.0f * tanhf(g[i] / 4) / (1 + expf(-g[i])) * 25.0f * tanhf(u[i] / 25)); } }
        }
        printf("hdebug: %zu / %zu h values differ\n", bad, n);
        return 0;
    }
    if (mode == "validate") {   // agent A's dumps: x.bin ids.bin g.bin u.bin h.bin dn.bin
        std::string dir = argv[6];
        auto X = slurp(dir + "/x.bin"), I = slurp(dir + "/ids.bin"), G = slurp(dir + "/g.bin"), U = slurp(dir + "/u.bin"), H = slurp(dir + "/h.bin"), DN = slurp(dir + "/dn.bin");
        const int nc = (int) (X.size() / (D * 4)), K = 16;
        int ncase = argc > 7 ? std::min(nc, atoi(argv[7])) : nc;
        long tot[4] = {0, 0, 0, 0}, ok[4] = {0, 0, 0, 0}; long rowsok = 0, rowstot = 0;
        std::vector<float> dg(K * F), du(K * F), dh(K * F);
        long hist[8] = {0};
        for (int c = 0; c < ncase; c++) {
            memcpy(en.x, X.data() + (size_t) c * D * 4, D * 4);
            memcpy(en.ids, I.data() + (size_t) c * K * 4, K * 4);
            en.run(K);
            en.q.memcpy(dg.data(), en.gbuf, K * F * 4); en.q.memcpy(du.data(), en.ubuf, K * F * 4); en.q.memcpy(dh.data(), en.hbuf, K * F * 4).wait();
            const uint32_t * rg = (const uint32_t *) (G.data() + (size_t) c * K * F * 4), * ru = (const uint32_t *) (U.data() + (size_t) c * K * F * 4);
            const uint32_t * rh = (const uint32_t *) (H.data() + (size_t) c * K * F * 4), * rd = (const uint32_t *) (DN.data() + (size_t) c * K * D * 4);
            int cbad[4] = {0, 0, 0, 0};
            for (int i = 0; i < K * F; i++) { tot[0]++; ok[0] += rg[i] == k3_f2u(dg[i]); tot[1]++; ok[1] += ru[i] == k3_f2u(du[i]); tot[2]++; ok[2] += rh[i] == k3_f2u(dh[i]);
                cbad[0] += rg[i] != k3_f2u(dg[i]); cbad[1] += ru[i] != k3_f2u(du[i]); cbad[2] += rh[i] != k3_f2u(dh[i]); }
            if (getenv("K3X_HFROM_DUMP")) { en.q.memcpy(en.hbuf, H.data() + (size_t) c * K * F * 4, K * F * 4).wait(); en.run_from_h(K); }
            for (int e = 0; e < K; e++) {
                bool rowok = true;
                for (int i = 0; i < D; i++) {
                    const uint32_t a = rd[e * D + i], b = k3_f2u(en.rows[e * D + i]);
                    tot[3]++; if (a == b) ok[3]++; else { rowok = false; cbad[3]++;
                        long dd = labs((long) (int32_t) a - (long) (int32_t) b); int bk = dd <= 1 ? 1 : dd <= 4 ? 2 : dd <= 16 ? 3 : dd <= 256 ? 4 : 5; hist[bk]++; }
                }
                rowstot++; rowsok += rowok;
            }
            if (getenv("K3X_SHOW") && (cbad[0] || cbad[1] || cbad[2] || cbad[3])) printf("  case %d: bad g %d u %d h %d dn %d\n", c, cbad[0], cbad[1], cbad[2], cbad[3]);
        }
        printf("validate %s (%d cases x %d experts): g %ld/%ld  u %ld/%ld  h %ld/%ld  dn %ld/%ld values bit-identical; whole expert rows exact %ld/%ld\n",
               dir.c_str(), ncase, K, ok[0], tot[0], ok[1], tot[1], ok[2], tot[2], ok[3], tot[3], rowsok, rowstot);
        printf("  dn ulp diff histogram: 1:%ld <=4:%ld <=16:%ld <=256:%ld more:%ld\n", hist[1], hist[2], hist[3], hist[4], hist[5]);
        return 0;
    }
    if (mode == "bench") {
        std::vector<int> ks; for (char * s = strtok(argv[6], ","); s; s = strtok(nullptr, ",")) ks.push_back(atoi(s));
        int iters = atoi(argv[7]);
        for (int i = 0; i < D; i++) en.x[i] = nd(rng);
        for (int k : ks) {
            std::vector<double> t;
            for (int it = 0; it < iters + 5; it++) {
                for (int i = 0; i < k; i++) en.ids[i] = rng() % E;
                double t0 = now_ms(); en.run(k); double dt = now_ms() - t0;
                if (it >= 5) t.push_back(dt);
            }
            std::sort(t.begin(), t.end());
            double p50 = t[t.size() / 2], p90 = t[(size_t) (t.size() * 0.9)];
            if (en.prof) { printf("  kernels (ms avg): quant %.3f gate/up %.3f act+quant %.3f down %.3f\n", en.kt[0] / en.nrun, en.kt[1] / en.nrun, en.kt[2] / en.nrun, en.kt[3] / en.nrun); for (double & v : en.kt) v = 0; en.nrun = 0; }
            printf("k=%2d: p50 %.3f ms  p90 %.3f ms  min %.3f ms  -> %.1f GB/s (p50)\n", k, p50, p90, t[0], k * en.rec / p50 / 1e6);
            fflush(stdout);
        }
        return 0;
    }
    if (mode == "serve") {
        int port = atoi(argv[6]);
        int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
        if (bind(ls, (sockaddr *) &a, sizeof a) || listen(ls, 4)) { perror("bind"); return 1; }
        en.spin = true; for (int i = 0; i < D; i++) en.x[i] = 0.1f; en.ids[0] = 0; en.run(1); // warm-up / JIT
        fprintf(stderr, "serving on %d (busy-poll)\n", port);
        auto rd = [](int fd, void * p, size_t n) { char * c = (char *) p; while (n) { ssize_t r = recv(fd, c, n, MSG_DONTWAIT); if (r == 0) return false; if (r < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK) continue; return false; } c += r; n -= r; } return true; };
        auto wr = [](int fd, const void * p, size_t n) { const char * c = (const char *) p; while (n) { ssize_t r = send(fd, c, n, 0); if (r <= 0) return false; c += r; n -= r; } return true; };
        for (;;) {
            int c = accept(ls, nullptr, nullptr);
            setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            int32_t hdr[17];
            while (rd(c, hdr, 17 * 4) && rd(c, en.x, D * 4)) {
                int k = hdr[0];
                if (k < 1 || k > 16) break;
                for (int i = 0; i < k; i++) en.ids[i] = ((hdr[1 + i] % E) + E) % E;
                en.run(k);
                if (!wr(c, en.rows, (size_t) k * D * 4)) break;
            }
            close(c);
        }
    }
    return 0;
}
