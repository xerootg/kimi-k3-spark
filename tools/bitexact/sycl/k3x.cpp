// k3x -- standalone bit-exact SYCL routed-expert FFN for Kimi K3 (one decode token), emulating the Spark's ggml-cuda kernels.
// usage:
//   k3x RECORDS N_RECORDS GATE_TYPE DOWN_TYPE check [N]            device vs host reference on random ids/x
//   k3x RECORDS N_RECORDS GATE_TYPE DOWN_TYPE bench k1,k2,.. ITERS  timing (p50/p90, GB/s)
//   k3x RECORDS N_RECORDS GATE_TYPE DOWN_TYPE serve PORT            rrclient/moebench wire protocol, busy-polling
//   RECORDS = "synthetic" makes random IQ records in memory.
#include <sycl/sycl.hpp>
#ifndef K3X_IB_DP4A
#define K3X_IB_DP4A 1
#endif
#ifndef K3X_DP16
#define K3X_DP16 0
#endif
#ifndef K3X_GRID_SLM
#define K3X_GRID_SLM 1
#endif
#ifndef K3X_FUSE_QX
#define K3X_FUSE_QX 0   // fused q8_1 of x per work-group measured slower (A/B on nuc2)
#endif
#ifndef K3X_DPF
#define K3X_DPF 0
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <cerrno>
#include <csignal>
#include <ctime>
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
#if defined(__SYCL_DEVICE_ONLY__) && K3X_IB_DP4A
// IGC's native DP4A (Xe-LP and later): c + sum of the four signed byte products, exact, no saturation.
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b, bool sat);
#endif
static inline int dp4a_dev(int a, int b, int c) {
#if defined(__SYCL_DEVICE_ONLY__) && K3X_DPF
    // byte products and partial sums are integers below 2^24 in magnitude, so fp32 FMAs are exact here
    float f = (float) c;
    f = sycl::fma((float) (int8_t) a, (float) (int8_t) b, f);
    f = sycl::fma((float) (int8_t) (a >> 8), (float) (int8_t) (b >> 8), f);
    f = sycl::fma((float) (int8_t) (a >> 16), (float) (int8_t) (b >> 16), f);
    f = sycl::fma((float) (int8_t) (a >> 24), (float) (int8_t) (b >> 24), f);
    return (int) f;
#elif defined(__SYCL_DEVICE_ONLY__) && K3X_DP16
    // this iGPU has no DP4A in IGC (the builtin lowers to 32x32 mul+mach pairs); byte products fit in 16 bits,
    // so do them as word multiplies (exact: |int8 * int8| <= 16384)
    const int16_t a0 = (int8_t) a, a1 = (int8_t) (a >> 8), a2 = (int8_t) (a >> 16), a3 = (int8_t) (a >> 24);
    const int16_t b0 = (int8_t) b, b1 = (int8_t) (b >> 8), b2 = (int8_t) (b >> 16), b3 = (int8_t) (b >> 24);
    const int16_t p0 = (int16_t) (a0 * b0), p1 = (int16_t) (a1 * b1), p2 = (int16_t) (a2 * b2), p3 = (int16_t) (a3 * b3);
    return c + ((int) p0 + (int) p1) + ((int) p2 + (int) p3);
#elif defined(__SYCL_DEVICE_ONLY__) && K3X_IB_DP4A
    return __builtin_IB_dp4a_ss(c, a, b, false);
#elif defined(__SYCL_DEVICE_ONLY__) && EXP == 5
    return (a ^ b) + c;
#elif defined(__SYCL_DEVICE_ONLY__)
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
// Transposed q8_1 activation layout for coalesced loads: for 256-block b (lane) and CUDA sub-lane l, word j of q8
// block 8b+l lives at QT[(l*8 + j)*16 + b]; its (d, s) floats at DT[l*16 + b]. Same values, only the layout differs.
static inline void store_V(const k3_q8_1 & blk, int qi, int * QV, float * DV) {
    for (int j = 0; j < 8; j++) { int w; memcpy(&w, blk.qs + 4 * j, 4); QV[qi * 8 + j] = w; }
    DV[qi * 2] = k3_h2f(blk.d); DV[qi * 2 + 1] = k3_h2f(blk.s);
}
static inline void store_T(const k3_q8_1 & blk, int qi, int * QT, sycl::float2 * DT) {
    const int b = qi / 8, l = qi % 8;
    for (int j = 0; j < 8; j++) { int w; memcpy(&w, blk.qs + 4 * j, 4); QT[(l * 8 + j) * 16 + b] = w; }
    DT[l * 16 + b] = sycl::float2(k3_h2f(blk.d), k3_h2f(blk.s));
}
static inline void blkT_iq1_s(const uint8_t * blk, const int * QT, const sycl::float2 * DT, int b, const uint32_t * grid, float * out) {
    const uint16_t * w = (const uint16_t *) blk;
    const uint16_t dh = w[0];
    const float dhf = h2f_dev(dh);
#pragma unroll
    for (int l = 0; l < 8; l++) {
        const uint32_t qsp = (uint32_t) w[1 + 2 * l] | ((uint32_t) w[2 + 2 * l] << 16);
        const int qh = w[17 + l];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#if EXP == 3
            const int g = (int) (qsp >> j) ^ qh;
#else
            const int g = (int) grid[((qsp >> (8 * j)) & 0xff) | (((qh >> (3 * j)) & 7) << 8)];
#endif
            sumi = dp4a_dev(g & 0x0F0F0F0F, QT[(l * 8 + 2 * j) * 16 + b], sumi);
            sumi = dp4a_dev((g >> 4) & 0x0F0F0F0F, QT[(l * 8 + 2 * j + 1) * 16 + b], sumi);
        }
#if EXP == 4
        out[l] = (float) sumi;
#else
        const sycl::float2 ds = DT[l * 16 + b];
        out[l] = iq1s_tail(dhf, qh, sumi, ds.x(), ds.y());
#endif
    }
}
static inline void blkT_iq2_xxs(const uint8_t * blk, const int * QT, const sycl::float2 * DT, int b, const uint64_t * grid, float * out) {
    const uint16_t * w = (const uint16_t *) blk;
    const float dhf = h2f_dev(w[0]);
#pragma unroll
    for (int l = 0; l < 8; l++) {
        const uint32_t q2 = (uint32_t) w[1 + 4 * l] | ((uint32_t) w[2 + 4 * l] << 16);
        const uint32_t aux32 = (uint32_t) w[3 + 4 * l] | ((uint32_t) w[4 + 4 * l] << 16);
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
            sumi = dp4a_dev((int) g0, QT[(l * 8 + 2 * j) * 16 + b], sumi);
            sumi = dp4a_dev((int) g1, QT[(l * 8 + 2 * j + 1) * 16 + b], sumi);
        }
        const int ls = (int) (aux32 >> 27 | 1);
        out[l] = iq2xxs_tail(dhf, sumi * ls / 8, DT[l * 16 + b].x());
    }
}

// ---- v3 building blocks: one CUDA lane-product p_t for (row-block w, sub-lane l), activation via the transposed layout
// (QT/DT indexed by 256-block kb; uniform across the sub-group because all 16 lanes are at the same (kb, l)).
static inline float p_iq1_s(const uint16_t * w, int l, const int * QT, const sycl::float2 * DT, int kb, const uint32_t * grid) {
    const uint32_t qsp = (uint32_t) w[1 + 2 * l] | ((uint32_t) w[2 + 2 * l] << 16);
    const int qh = w[17 + l];
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
#if EXP == 9
        const int g = (int) (qsp >> j) ^ qh;
#else
        const int g = (int) grid[((qsp >> (8 * j)) & 0xff) | (((qh >> (3 * j)) & 7) << 8)];
#endif
#if EXP == 8
        sumi = dp4a_dev(g & 0x0F0F0F0F, l * 8 + 2 * j + kb, sumi);
        sumi = dp4a_dev((g >> 4) & 0x0F0F0F0F, l * 8 + 2 * j + 1 + kb, sumi);
#else
        sumi = dp4a_dev(g & 0x0F0F0F0F, QT[(l * 8 + 2 * j) * 16 + kb], sumi);
        sumi = dp4a_dev((g >> 4) & 0x0F0F0F0F, QT[(l * 8 + 2 * j + 1) * 16 + kb], sumi);
#endif
    }
#if EXP == 8
    const sycl::float2 ds(0.5f * kb, 0.25f * l);
#else
    const sycl::float2 ds = DT[l * 16 + kb];
#endif
    return iq1s_tail(h2f_dev(w[0]), qh, sumi, ds.x(), ds.y());
}
static inline float p_iq2_xxs(const uint16_t * w, int l, const int * QT, const sycl::float2 * DT, int kb, const uint64_t * grid) {
    const uint32_t q2 = (uint32_t) w[1 + 4 * l] | ((uint32_t) w[2 + 4 * l] << 16);
    const uint32_t aux32 = (uint32_t) w[3 + 4 * l] | ((uint32_t) w[4 + 4 * l] << 16);
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
        sumi = dp4a_dev((int) g0, QT[(l * 8 + 2 * j) * 16 + kb], sumi);
        sumi = dp4a_dev((int) g1, QT[(l * 8 + 2 * j + 1) * 16 + kb], sumi);
    }
    const int ls = (int) (aux32 >> 27 | 1);
    return iq2xxs_tail(h2f_dev(w[0]), sumi * ls / 8, DT[l * 16 + kb].x());
}
// One whole output row per work-item. CUDA's per-row order: s_x = ((p_x + p_{x+32}) + p_{x+64}) + p_{x+96} with
// x = 8*kb0 + l, p for 256-block kb = kb0 + 4i; idle CUDA threads (kb >= nblk) contribute +0.0f; then the xor tree.
template <int T, int NBLK>
static inline float row_v3(const uint8_t * rowp, const int * QT, const sycl::float2 * DT, const uint32_t * g1, const uint64_t * g2) {
    constexpr int BB = T == T_IQ1_S ? 50 : 66;
    constexpr int NW = BB / 2;
    float s[32];
#pragma unroll
    for (int kb0 = 0; kb0 < 4; kb0++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int kb = kb0 + 4 * i;
            if (kb < NBLK) {
                uint16_t w[NW];
                const uint16_t * src = (const uint16_t *) (rowp + kb * BB);
#pragma unroll
                for (int u = 0; u < NW; u++) w[u] = src[u];
#pragma unroll
                for (int l = 0; l < 8; l++) {
#if EXP == 6
                    const float p = (float) (w[l] + w[l + 8] + w[l + 16]);
#elif EXP == 7
                    const float p = (float) (w[l] + w[l + 8] + w[l + 16] + QT[l * 128 + kb] + g1[w[l] & 2047]);
#else
                    const float p = T == T_IQ1_S ? p_iq1_s(w, l, QT, DT, kb, g1) : p_iq2_xxs(w, l, QT, DT, kb, g2);
#endif
                    s[kb0 * 8 + l] = i == 0 ? p : s[kb0 * 8 + l] + p;
                }
            } else {
#pragma unroll
                for (int l = 0; l < 8; l++) s[kb0 * 8 + l] = s[kb0 * 8 + l] + 0.0f;
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int x = 0; x < off; x++) s[x] = s[x] + s[x + off];
    return s[0];
}

// ---- v4: one row per work-item like v3, but with far fewer memory messages (the LSC message rate was the limit):
//  * activation: the q8 words of 256-block kb are 64 contiguous ints (natural q8_1 block order, block 8kb+l, word j at
//    kb*64 + 8l + j); lane i loads ints 4i..4i+3 as one coalesced uint4 and values are broadcast with constant lanes;
//    (d, s) floats likewise (16 floats per kb, lane i holds float i).
//  * weights: the 50/66-byte block is read as 13/17 aligned dwords (row and record offsets are multiples of 4; the
//    block's 2-byte misalignment for odd kb is a compile-time shift).
//  * grid tables in SLM.
// Arithmetic per p_t is unchanged (same integer dot, same float tail); the row reduction is row_v3's exact order.
template <int T> struct v4w { static constexpr int BB = T == T_IQ1_S ? 50 : 66; static constexpr int NDW = (BB + 2 + 3) / 4; };
template <int T, int KB>
static inline void load_block_v4(const uint8_t * rowp, uint16_t * w) {
    constexpr int BB = v4w<T>::BB, NDW = v4w<T>::NDW, OFF = (KB * BB) % 4;   // OFF is 0 or 2
    const uint32_t * src = (const uint32_t *) (rowp + KB * BB - OFF);
    uint32_t d[NDW];
#pragma unroll
    for (int u = 0; u < NDW; u++) d[u] = src[u];
#pragma unroll
    for (int u = 0; u < BB / 2; u++) { const int byte = OFF + 2 * u; w[u] = (uint16_t) (d[byte / 4] >> (8 * (byte % 4))); }
}
template <int T, int NBLK>
static inline float row_v4(sycl::sub_group sg, const uint8_t * rowp, const int * QV, const float * DV, const uint32_t * g1, const uint64_t * g2) {
    const int lane = sg.get_local_linear_id();
    float s[32];
    auto do_kb = [&](auto kbc) {
        constexpr int kb = decltype(kbc)::value;
        constexpr int kb0 = kb % 4, i = kb / 4;
        if constexpr (kb < NBLK) {
            uint16_t w[v4w<T>::BB / 2];
            load_block_v4<T, kb>(rowp, w);
            const sycl::int4 q4 = *(const sycl::int4 *) (QV + kb * 64 + 4 * lane);
            const float dsl = DV[kb * 16 + lane];
#pragma unroll
            for (int l = 0; l < 8; l++) {
                int qv[8];
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int idx = l * 8 + j;   // lane idx/4, component idx%4 (all compile-time after unrolling)
                    const int c = (idx % 4 == 0) ? q4.x() : (idx % 4 == 1) ? q4.y() : (idx % 4 == 2) ? q4.z() : q4.w();
                    qv[j] = sycl::group_broadcast(sg, c, idx / 4);
                }
                const float dx = sycl::group_broadcast(sg, dsl, 2 * l), dy = sycl::group_broadcast(sg, dsl, 2 * l + 1);
                float p;
                if constexpr (T == T_IQ1_S) {
                    const uint32_t qsp = (uint32_t) w[1 + 2 * l] | ((uint32_t) w[2 + 2 * l] << 16);
                    const int qh = w[17 + l];
                    int sumi = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int g = (int) g1[((qsp >> (8 * j)) & 0xff) | (((qh >> (3 * j)) & 7) << 8)];
                        sumi = dp4a_dev(g & 0x0F0F0F0F, qv[2 * j], sumi);
                        sumi = dp4a_dev((g >> 4) & 0x0F0F0F0F, qv[2 * j + 1], sumi);
                    }
                    p = iq1s_tail(h2f_dev(w[0]), qh, sumi, dx, dy);
                } else {
                    const uint32_t q2 = (uint32_t) w[1 + 4 * l] | ((uint32_t) w[2 + 4 * l] << 16);
                    const uint32_t aux32 = (uint32_t) w[3 + 4 * l] | ((uint32_t) w[4 + 4 * l] << 16);
                    int sumi = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint64_t gp = g2[(q2 >> (8 * j)) & 0xff];
                        const uint32_t v = (aux32 >> (7 * j)) & 0xff;
                        const uint32_t sgn = (v ^ ((sycl::popcount(v) & 1) << 7)) * 0x01010101u;
                        const uint32_t m0 = (((sgn & 0x08040201u) + 0x7f7f7f7fu) & 0x80808080u) >> 7;
                        const uint32_t m1 = (((sgn & 0x80402010u) + 0x7f7f7f7fu) & 0x80808080u) >> 7;
                        const uint32_t ga = ((uint32_t) gp ^ (m0 * 0xffu)) + m0;
                        const uint32_t gb = ((uint32_t) (gp >> 32) ^ (m1 * 0xffu)) + m1;
                        sumi = dp4a_dev((int) ga, qv[2 * j], sumi);
                        sumi = dp4a_dev((int) gb, qv[2 * j + 1], sumi);
                    }
                    const int ls = (int) (aux32 >> 27 | 1);
                    p = iq2xxs_tail(h2f_dev(w[0]), sumi * ls / 8, dx);
                }
                s[kb0 * 8 + l] = i == 0 ? p : s[kb0 * 8 + l] + p;
            }
        } else {
#pragma unroll
            for (int l = 0; l < 8; l++) s[kb0 * 8 + l] = s[kb0 * 8 + l] + 0.0f;
        }
    };
    // CUDA order per s_x: kb0 first, then kb0+4, kb0+8, kb0+12
    [&]<int... I>(std::integer_sequence<int, I...>) { (do_kb(std::integral_constant<int, (I % 4) * 4 + I / 4>{}), ...); }(std::make_integer_sequence<int, 16>{});
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int x = 0; x < off; x++) s[x] = s[x] + s[x + off];
    return s[0];
}

template <int T, int NBLK>
static inline float row_v5(const uint32_t * mat, int rg, int lane, const int * QT, const sycl::float2 * DT, const uint32_t * g1, const uint64_t * g2) {
    constexpr int NDW = T == T_IQ1_S ? 13 : 17, NW = (T == T_IQ1_S ? 50 : 66) / 2;
    float s[32];
#pragma unroll
    for (int kb0 = 0; kb0 < 4; kb0++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int kb = kb0 + 4 * i;
            if (kb < NBLK) {
                const uint32_t * bp = mat + ((size_t) (rg * NBLK + kb) * NDW) * 16 + lane;
                uint32_t d[NDW];
#if EXP == 10
#pragma unroll
                for (int u = 0; u < NDW; u++) d[u] = (uint32_t) (lane * 77 + u * kb + rg);
#else
#pragma unroll
                for (int u = 0; u < NDW; u++) d[u] = bp[u * 16];
#endif
                uint16_t w[NW];
#pragma unroll
                for (int u = 0; u < NW; u++) w[u] = (uint16_t) (d[u / 2] >> (16 * (u & 1)));
#pragma unroll
                for (int l = 0; l < 8; l++) {
                    const float p = T == T_IQ1_S ? p_iq1_s(w, l, QT, DT, kb, g1) : p_iq2_xxs(w, l, QT, DT, kb, g2);
                    s[kb0 * 8 + l] = i == 0 ? p : s[kb0 * 8 + l] + p;
                }
            } else {
#pragma unroll
                for (int l = 0; l < 8; l++) s[kb0 * 8 + l] = s[kb0 * 8 + l] + 0.0f;
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int x = 0; x < off; x++) s[x] = s[x] + s[x + off];
    return s[0];
}

static inline float pq_iq1_s(const uint16_t * w, int l, const int * qv, float dx, float dy, const uint32_t * grid) {
    const uint32_t qsp = (uint32_t) w[1 + 2 * l] | ((uint32_t) w[2 + 2 * l] << 16);
    const int qh = w[17 + l];
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int g = (int) grid[((qsp >> (8 * j)) & 0xff) | (((qh >> (3 * j)) & 7) << 8)];
        sumi = dp4a_dev(g & 0x0F0F0F0F, qv[2 * j], sumi);
        sumi = dp4a_dev((g >> 4) & 0x0F0F0F0F, qv[2 * j + 1], sumi);
    }
    return iq1s_tail(h2f_dev(w[0]), qh, sumi, dx, dy);
}
static inline float pq_iq2_xxs(const uint16_t * w, int l, const int * qv, float dx, const uint64_t * grid) {
    const uint32_t q2 = (uint32_t) w[1 + 4 * l] | ((uint32_t) w[2 + 4 * l] << 16);
    const uint32_t aux32 = (uint32_t) w[3 + 4 * l] | ((uint32_t) w[4 + 4 * l] << 16);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint64_t gp = grid[(q2 >> (8 * j)) & 0xff];
        const uint32_t v = (aux32 >> (7 * j)) & 0xff;
        const uint32_t sgn = (v ^ ((sycl::popcount(v) & 1) << 7)) * 0x01010101u;
        const uint32_t m0 = (((sgn & 0x08040201u) + 0x7f7f7f7fu) & 0x80808080u) >> 7;
        const uint32_t m1 = (((sgn & 0x80402010u) + 0x7f7f7f7fu) & 0x80808080u) >> 7;
        const uint32_t ga = ((uint32_t) gp ^ (m0 * 0xffu)) + m0;
        const uint32_t gb = ((uint32_t) (gp >> 32) ^ (m1 * 0xffu)) + m1;
        sumi = dp4a_dev((int) ga, qv[2 * j], sumi);
        sumi = dp4a_dev((int) gb, qv[2 * j + 1], sumi);
    }
    const int ls = (int) (aux32 >> 27 | 1);
    return iq2xxs_tail(h2f_dev(w[0]), sumi * ls / 8, dx);
}
// v6 = v5 with the activation of block kb loaded once per sub-group (lane i: ints 4i..4i+3 and float i of the natural
// q8 layout QV/DV) and handed out by constant-lane shuffles instead of 9 uniform loads per p_t.
template <int T, int NBLK>
static inline float row_v6(sycl::sub_group sg, const uint32_t * mat, int rg, int lane, const int * QV, const float * DV, const uint32_t * g1, const uint64_t * g2) {
    constexpr int NDW = T == T_IQ1_S ? 13 : 17, NW = (T == T_IQ1_S ? 50 : 66) / 2;
    float s[32];
#pragma unroll
    for (int kb0 = 0; kb0 < 4; kb0++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int kb = kb0 + 4 * i;
            if (kb < NBLK) {
                const uint32_t * bp = mat + ((size_t) (rg * NBLK + kb) * NDW) * 16 + lane;
                uint32_t d[NDW];
#pragma unroll
                for (int u = 0; u < NDW; u++) d[u] = bp[u * 16];
                const int sl = (int) sg.get_local_linear_id();
                const int * qp = QV + kb * 64 + 4 * sl;
                const int qa = qp[0], qb = qp[1], qc = qp[2], qd = qp[3];
                const float dl = DV[kb * 16 + sl];
                uint16_t w[NW];
#pragma unroll
                for (int u = 0; u < NW; u++) w[u] = (uint16_t) (d[u / 2] >> (16 * (u & 1)));
#pragma unroll
                for (int l = 0; l < 8; l++) {
                    int qv[8];
#pragma unroll
                    for (int j = 0; j < 8; j += 4) {
                        const int src = (l * 8 + j) / 4;
                        qv[j + 0] = sycl::group_broadcast(sg, qa, src);
                        qv[j + 1] = sycl::group_broadcast(sg, qb, src);
                        qv[j + 2] = sycl::group_broadcast(sg, qc, src);
                        qv[j + 3] = sycl::group_broadcast(sg, qd, src);
                    }
                    float dx = sycl::group_broadcast(sg, dl, 2 * l), dy = sycl::group_broadcast(sg, dl, 2 * l + 1);
#if EXP == 11
                    for (int j = 0; j < 8; j++) qv[j] = QV[kb * 64 + l * 8 + j];
                    dx = DV[kb * 16 + 2 * l]; dy = DV[kb * 16 + 2 * l + 1];
#endif
                    const float p = T == T_IQ1_S ? pq_iq1_s(w, l, qv, dx, dy, g1) : pq_iq2_xxs(w, l, qv, dx, g2);
                    s[kb0 * 8 + l] = i == 0 ? p : s[kb0 * 8 + l] + p;
                }
            } else {
#pragma unroll
                for (int l = 0; l < 8; l++) s[kb0 * 8 + l] = s[kb0 * 8 + l] + 0.0f;
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int x = 0; x < off; x++) s[x] = s[x] + s[x + off];
    return s[0];
}

// v7 layout without padding blocks: per group of 4 rows, i = 0..2 hold 16 lanes (kb0 0..3 x 4 rows) x NDW dwords, and for
// 14 blocks i = 3 holds only kb 12, 13 (8 lanes). Dword u of lane L for i<3 at ((i*NDW + u)*16 + L), for i=3 at 48*NDW + u*8 + L.
static inline size_t v7_off(int nblk, int ndw, int rg4, int i, int u, int lane) {
    const size_t g = (size_t) rg4 * (nblk * 4) * ndw;
    return i < 3 ? g + ((size_t) i * ndw + u) * 16 + lane : g + (size_t) 48 * ndw + (size_t) u * 8 + lane;
}
template <int T, int NBLK>
static inline float row_v7(sycl::sub_group sg, const uint32_t * mat, int rg4, int lane, const int * QT, const sycl::float2 * DT, const uint32_t * g1, const uint64_t * g2) {
    constexpr int NDW = T == T_IQ1_S ? 13 : 17, NW = (T == T_IQ1_S ? 50 : 66) / 2;
    const int q = lane / 4;   // kb0
    float s[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int kb = q + 4 * i;
        if (kb < NBLK) {
            uint32_t d[NDW];
            const uint32_t * bp = mat + v7_off(NBLK, NDW, rg4, i, 0, lane);
            const int st = i < 3 ? 16 : 8;
#pragma unroll
            for (int u = 0; u < NDW; u++) d[u] = bp[u * st];
            uint16_t w[NW];
#pragma unroll
            for (int u = 0; u < NW; u++) w[u] = (uint16_t) (d[u / 2] >> (16 * (u & 1)));
#pragma unroll
            for (int l = 0; l < 8; l++) {
                const float p = T == T_IQ1_S ? p_iq1_s(w, l, QT, DT, kb, g1) : p_iq2_xxs(w, l, QT, DT, kb, g2);
                s[l] = i == 0 ? p : s[l] + p;
            }
        } else {
#pragma unroll
            for (int l = 0; l < 8; l++) s[l] = s[l] + 0.0f;
        }
    }
    // xor tree over x = 8*kb0 + l: off 16 = lane + 8 (kb0 + 2), off 8 = lane + 4 (kb0 + 1), then inside the lane
#pragma unroll
    for (int l = 0; l < 8; l++) s[l] = s[l] + sycl::select_from_group(sg, s[l], (lane + 8) & 15);
#pragma unroll
    for (int l = 0; l < 8; l++) s[l] = s[l] + sycl::select_from_group(sg, s[l], (lane + 4) & 15);
#pragma unroll
    for (int l = 0; l < 4; l++) s[l] = s[l] + s[l + 4];
    s[0] = s[0] + s[2]; s[1] = s[1] + s[3];
    return s[0] + s[1];
}
static inline void blk_any(int t, const uint8_t * blk, const k3_q8_1 * y8, const sycl::float2 * dsf, const uint32_t * g1, const uint64_t * g2, float * out) {
    if (t == T_IQ1_S) blk_iq1_s(blk, y8, dsf, g1, out); else blk_iq2_xxs(blk, y8, dsf, g2, out);
}

static constexpr int D = K3_D, F = K3_F;
#ifndef MV5_WG
#define MV5_WG 512
#endif
#ifndef V7
#define V7 1
#endif
#ifndef V6
#define V6 0   // v6 (constant-lane shuffles) gave wrong rows on this IGC and no speed-up: off
#endif
#ifndef V5
#define V5 1
#endif
#ifndef MV4
#define MV4 1
#endif
#ifndef MV4_WG
#define MV4_WG 256
#endif
#ifndef MV3
#define MV3 1
#endif
#ifndef MV3_WG
#define MV3_WG 64
#endif
#ifndef MV2
#define MV2 1
#endif
#ifndef MV2_SG
#define MV2_SG 8
#endif
#ifndef MV2_RPS
#define MV2_RPS 2
#endif
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

// ---- v5 weight layout (repacked once at load; pure data movement, values unchanged): per matrix, per group of 16 rows,
// per 256-block kb, NDW dwords x 16 rows with the row fastest, so dword u of block kb for the 16 rows of a sub-group
// is one contiguous 64-byte line. NDW = 13 for IQ1_S (50 B + 2 pad), 17 for IQ2_XXS (66 B + 2 pad).
static inline int v5_ndw(int t) { return (k3_blk_bytes(t) + 3) / 4; }
static inline size_t v5_mat_bytes(int t, int nrows, int ncols) { return (size_t) (nrows / 16) * (ncols / K3_QK) * v5_ndw(t) * 64; }
static void v5_repack_matrix(const uint8_t * src, uint8_t * dst, int t, int nrows, int ncols) {
    const int bb = k3_blk_bytes(t), ndw = v5_ndw(t), nblk = ncols / K3_QK;
    const size_t rb = (size_t) nblk * bb;
    uint32_t * o = (uint32_t *) dst;
    for (int rg = 0; rg < nrows / 16; rg++)
        for (int kb = 0; kb < nblk; kb++)
            for (int u = 0; u < ndw; u++)
                for (int ln = 0; ln < 16; ln++) {
                    uint8_t b4[4] = {0, 0, 0, 0};
                    for (int c = 0; c < 4; c++) if (u * 4 + c < bb) b4[c] = src[(size_t) (rg * 16 + ln) * rb + (size_t) kb * bb + u * 4 + c];
                    uint32_t v; memcpy(&v, b4, 4);
                    o[(((size_t) rg * nblk + kb) * ndw + u) * 16 + ln] = v;
                }
}

// ---- v7 layout: 4 lanes per row (lane q = kb0 handles blocks kb0, kb0+4, kb0+8, kb0+12), 4 rows per sub-group.
// Per group of 4 rows, per i (0..3), per dword u: 16 dwords ordered (kb0, row) so one load is one 64-byte line.
// Blocks kb >= nblk (i = 3, kb0 >= 2) are zero padding and never used. Pure data movement.
static inline size_t v7_mat_bytes(int t, int nrows, int ncols) { return (size_t) nrows * (ncols / K3_QK) * v5_ndw(t) * 4; }
static void v7_repack_matrix(const uint8_t * src, uint8_t * dst, int t, int nrows, int ncols) {
    const int bb = k3_blk_bytes(t), ndw = v5_ndw(t), nblk = ncols / K3_QK;
    const size_t rb = (size_t) nblk * bb;
    uint32_t * o = (uint32_t *) dst;
    for (int rg = 0; rg < nrows / 4; rg++)
        for (int i = 0; i < 4; i++)
            for (int u = 0; u < ndw; u++)
                for (int q = 0; q < 4; q++)
                    for (int j = 0; j < 4; j++) {
                        const int kb = q + 4 * i;
                        if (kb >= nblk) continue;
                        uint8_t b4[4] = {0, 0, 0, 0};
                        for (int c = 0; c < 4; c++) if (u * 4 + c < bb) b4[c] = src[(size_t) (rg * 4 + j) * rb + (size_t) kb * bb + u * 4 + c];
                        uint32_t v; memcpy(&v, b4, 4);
                        o[v7_off(nblk, ndw, rg, i, u, q * 4 + j)] = v;
                    }
}
struct RecPtrs { const uint8_t * p[16]; };

// K3X_TRACE=<ring entries> (serveidx, off by default): per-request timestamps (CLOCK_MONOTONIC_RAW ns) in a ring that is
// allocated at start, so the request path only stores integers. Written as TSV to K3X_TRACE_OUT (default
// /tmp/k3x-trace.tsv) on SIGUSR1 (while waiting for a request) and whenever a connection closes. The client's sequence
// number is header word 16 when k < 16 (the traced engine and rrclient RR_SEQ=1 put it there; other clients send 0).
// K3X_TRACE_PROF=1 also turns on queue profiling and records every command's submit/start/end (device clock, ns; read
// after the reply is sent). K3X_TRACE_FREQ=1 reads the GT's actual frequency (sysfs) before and after the GPU work.
// tools/bitexact/sycl/trace_report.py turns a dump (and the engine's) into a per-stage breakdown.
static inline uint64_t k3x_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return (uint64_t) t.tv_sec * 1000000000ull + (uint64_t) t.tv_nsec; }
constexpr int TR_K = 4;               // rows traced per request (production clients send k <= 2)
constexpr int TR_NC = 4 + 2 * TR_K;   // commands: 0 x to device, 1 q8_1(x), 2 gate/up, 3 SiTU + q8_1(h), 4+2e down e, 5+2e row e to host
struct k3x_tr {
    int64_t idx, seq; int32_t k, conn;
    uint64_t t_wait, t_first, t_hdr, t_rx, t_x, t_end, t_done;   // wait start, first byte, header, x received, x in host USM, queue drained, done
    uint64_t t_sub[TR_NC];                                       // each submission returned
    uint64_t t_row[TR_K], t_sent[TR_K];                          // row e in host memory (polled), row e written to the socket
    uint64_t d_sub[TR_NC], d_start[TR_NC], d_end[TR_NC];         // K3X_TRACE_PROF: device clock
    int32_t f_before, f_after;                                   // K3X_TRACE_FREQ: MHz
};
static k3x_tr * g_tr = nullptr; static size_t g_tr_n = 0; static int64_t g_tr_next = 0;
static bool g_tr_prof = false; static int g_tr_freq_fd = -1;
static volatile sig_atomic_t g_tr_dump = 0;
static void k3x_tr_sig(int) { g_tr_dump = 1; }
static int k3x_tr_freq() {
    if (g_tr_freq_fd < 0) return 0;
    char b[16] = {0}; return pread(g_tr_freq_fd, b, sizeof b - 1, 0) > 0 ? atoi(b) : -1;
}
static void k3x_tr_init() {
    const char * e = getenv("K3X_TRACE"); if (!e || !atoi(e)) return;
    g_tr_n = (size_t) std::max(1024, atoi(e) > 1 ? atoi(e) : 32768);
    g_tr = (k3x_tr *) calloc(g_tr_n, sizeof(k3x_tr));
    g_tr_prof = getenv("K3X_TRACE_PROF") != nullptr;
    if (getenv("K3X_TRACE_FREQ")) g_tr_freq_fd = open("/sys/class/drm/card0/gt_act_freq_mhz", O_RDONLY);
    signal(SIGUSR1, k3x_tr_sig);
    fprintf(stderr, "K3X_TRACE: %zu entries%s%s, SIGUSR1 dumps to %s\n", g_tr_n, g_tr_prof ? ", queue profiling" : "",
            g_tr_freq_fd >= 0 ? ", GT frequency" : "", getenv("K3X_TRACE_OUT") ? getenv("K3X_TRACE_OUT") : "/tmp/k3x-trace.tsv");
}
static k3x_tr * k3x_tr_begin() {   // before waiting for a request (off the request path)
    if (!g_tr) return nullptr;
    k3x_tr * r = &g_tr[(size_t) (g_tr_next % (int64_t) g_tr_n)];
    memset(r, 0, sizeof *r); r->idx = g_tr_next++; r->t_wait = k3x_ns();
    return r;
}
static void k3x_tr_dump() {
    g_tr_dump = 0;
    if (!g_tr) return;
    const char * p = getenv("K3X_TRACE_OUT") ? getenv("K3X_TRACE_OUT") : "/tmp/k3x-trace.tsv";
    FILE * f = fopen(p, "w"); if (!f) { perror(p); return; }
    fprintf(f, "idx\tseq\tconn\tk\tt_wait\tt_first\tt_hdr\tt_rx\tt_x");
    for (int i = 0; i < TR_NC; i++) fprintf(f, "\tt_sub%d", i);
    for (int i = 0; i < TR_K; i++) fprintf(f, "\tt_row%d", i);
    for (int i = 0; i < TR_K; i++) fprintf(f, "\tt_sent%d", i);
    fprintf(f, "\tt_end\tt_done\tf_before\tf_after");
    for (int i = 0; i < TR_NC; i++) fprintf(f, "\td_sub%d\td_start%d\td_end%d", i, i, i);
    fprintf(f, "\n");
    const int64_t n = std::min<int64_t>(g_tr_next, (int64_t) g_tr_n);
    for (int64_t j = g_tr_next - n; j < g_tr_next; j++) {
        const k3x_tr & r = g_tr[(size_t) (j % (int64_t) g_tr_n)];
        if (!r.t_first) continue;   // still waiting for it
        fprintf(f, "%lld\t%lld\t%d\t%d\t%llu\t%llu\t%llu\t%llu\t%llu", (long long) r.idx, (long long) r.seq, r.conn, r.k,
                (unsigned long long) r.t_wait, (unsigned long long) r.t_first, (unsigned long long) r.t_hdr, (unsigned long long) r.t_rx, (unsigned long long) r.t_x);
        for (int i = 0; i < TR_NC; i++) fprintf(f, "\t%llu", (unsigned long long) r.t_sub[i]);
        for (int i = 0; i < TR_K; i++) fprintf(f, "\t%llu", (unsigned long long) r.t_row[i]);
        for (int i = 0; i < TR_K; i++) fprintf(f, "\t%llu", (unsigned long long) r.t_sent[i]);
        fprintf(f, "\t%llu\t%llu\t%d\t%d", (unsigned long long) r.t_end, (unsigned long long) r.t_done, r.f_before, r.f_after);
        for (int i = 0; i < TR_NC; i++) fprintf(f, "\t%llu\t%llu\t%llu", (unsigned long long) r.d_sub[i], (unsigned long long) r.d_start[i], (unsigned long long) r.d_end[i]);
        fprintf(f, "\n");
    }
    fclose(f);
    fprintf(stderr, "K3X_TRACE: %lld requests written to %s\n", (long long) n, p);
}

struct Engine {
    sycl::queue q;
    int tg, td, E;
    size_t rbg, rbd, gsz, dsz, rec, pg = 0, pd = 0, prec = 0;   // prec = record bytes on the device (v5: repacked)   // row bytes, matrix bytes, record bytes
    uint8_t * W = nullptr;
    uint32_t * g1 = nullptr; uint64_t * g2 = nullptr;
    float * x = nullptr; int * ids = nullptr; float * rows = nullptr; const uint8_t ** recs = nullptr; bool ext_recs = false; bool spin = false;
    float * xd = nullptr, * rowsd = nullptr;   // device copies: explicit memcpy in/out (host USM read/written by kernels was not reliably coherent)   // host USM (zero copy on the iGPU)
    k3_q8_1 * yq = nullptr, * yq2 = nullptr; int * QT = nullptr, * QT2 = nullptr; int * QV = nullptr, * QV2 = nullptr; float * DV = nullptr, * DV2 = nullptr; sycl::float2 * DT = nullptr, * DT2 = nullptr; sycl::float2 * dsf = nullptr, * dsf2 = nullptr;
    float * gbuf = nullptr, * ubuf = nullptr, * hbuf = nullptr;
    k3_mufu_tabs mt{};
    int nwg = getenv("K3X_NWG") ? atoi(getenv("K3X_NWG")) : 56;
    bool prof = false; std::vector<sycl::event> ev; double kt[4] = {0, 0, 0, 0}; double kspan = 0; long nrun = 0;
    k3x_tr * tr = nullptr;   // K3X_TRACE: the request being served (set by serve_index, nullptr otherwise)

    static sycl::property_list props(bool p) {
        if (p) return sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
        return sycl::property_list{sycl::property::queue::in_order()};
    }
    static sycl::queue & shared_queue() {
        if (!g_queue) g_queue = new sycl::queue(sycl::gpu_selector_v, async_err, props(getenv("K3X_PROF") != nullptr || getenv("K3X_TRACE_PROF") != nullptr));
        return *g_queue;
    }
    Engine(int tg_, int td_, int E_) : q(shared_queue()), tg(tg_), td(td_), E(E_) {
        prof = getenv("K3X_PROF") != nullptr;
        rbg = (size_t) D / K3_QK * k3_blk_bytes(tg); rbd = (size_t) F / K3_QK * k3_blk_bytes(td);
        gsz = rbg * F; dsz = rbd * D; rec = 2 * gsz + dsz;
        pg = V7 ? v7_mat_bytes(tg, F, D) : v5_mat_bytes(tg, F, D); pd = V7 ? v7_mat_bytes(td, D, F) : v5_mat_bytes(td, D, F); prec = V5 ? 2 * pg + pd : rec;
        if (E > 0) W = sycl::malloc_device<uint8_t>(prec * E, q);
        g1 = sycl::malloc_device<uint32_t>(2048, q); g2 = sycl::malloc_device<uint64_t>(256, q);
        q.memcpy(g1, iq1s_grid_gpu, 2048 * 4); q.memcpy(g2, iq2xxs_grid, 256 * 8);
        x = sycl::malloc_host<float>(D, q); ids = sycl::malloc_host<int>(16, q); recs = sycl::malloc_host<const uint8_t *>(16, q); rows = sycl::malloc_host<float>(16 * D, q);
        xd = sycl::malloc_device<float>(D, q); rowsd = sycl::malloc_device<float>(16 * D, q);
        yq = sycl::malloc_device<k3_q8_1>(D / 32, q); yq2 = sycl::malloc_device<k3_q8_1>(16 * F / 32, q);
        dsf = sycl::malloc_device<sycl::float2>(D / 32, q); dsf2 = sycl::malloc_device<sycl::float2>(16 * F / 32, q);
        QT = sycl::malloc_device<int>(1024, q); QT2 = sycl::malloc_device<int>(16 * 1024, q); DT = sycl::malloc_device<sycl::float2>(128, q); DT2 = sycl::malloc_device<sycl::float2>(16 * 128, q);
        QV = sycl::malloc_device<int>(896, q); QV2 = sycl::malloc_device<int>(16 * 896, q); DV = sycl::malloc_device<float>(224, q); DV2 = sycl::malloc_device<float>(16 * 224, q);
        q.memset(QV, 0, 896 * 4); q.memset(QV2, 0, 16 * 896 * 4); q.memset(DV, 0, 224 * 4); q.memset(DV2, 0, 16 * 224 * 4);
        q.memset(QT, 0, 4096); q.memset(QT2, 0, 16 * 4096); q.memset(DT, 0, 128 * 8); q.memset(DT2, 0, 16 * 128 * 8);
        gbuf = sycl::malloc_device<float>(16 * F, q); ubuf = sycl::malloc_device<float>(16 * F, q); hbuf = sycl::malloc_device<float>(16 * F, q);
        q.wait();
        fprintf(stderr, "device: %s; record %zu B (gate/up %zu, down %zu), %d records = %.1f MB\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), rec, gsz, dsz, E, rec * E / 1e6);
    }

    void repack(const uint8_t * src, uint8_t * dst) const {   // one record, original -> device layout
        if (!V5) { memcpy(dst, src, rec); return; }
        if (V7) { v7_repack_matrix(src, dst, tg, F, D); v7_repack_matrix(src + gsz, dst + pg, tg, F, D); v7_repack_matrix(src + 2 * gsz, dst + 2 * pg, td, D, F); return; }
        v5_repack_matrix(src, dst, tg, F, D); v5_repack_matrix(src + gsz, dst + pg, tg, F, D); v5_repack_matrix(src + 2 * gsz, dst + 2 * pg, td, D, F);
    }
    void upload(const uint8_t * host, size_t e0, size_t n) {
        uint8_t * tmp = sycl::malloc_host<uint8_t>(prec, q);   // pinned staging (see serve_index)
        for (size_t i = 0; i < n; i++) { repack(host + i * rec, tmp); q.memcpy(W + (e0 + i) * prec, tmp, prec).wait(); }
        sycl::free(tmp, q);
    }

    template <int T, int NBLK>
    sycl::event matvec5_t(int k, int nmat, int nrows, size_t moff, size_t mstride,
                          const int * qt, const sycl::float2 * dt, int estride, float * out0, float * out1,
                          const int * qv = nullptr, const float * dv = nullptr, int ebase = 0, const float * xsrc = nullptr, int xnb = 0) {
        const uint32_t * g1_ = g1; const uint64_t * g2_ = g2; const k3_mufu_tabs mt_ = mt;
        RecPtrs recs_; for (int i = 0; i < 16; i++) recs_.p[i] = recs[i];
        const long nitems = (long) k * nmat * nrows * (V7 ? 4 : 1);
        const int wg = MV5_WG;   // divides 3072 and 3584, so a work-group never spans two (expert, matrix) pairs
        return q.submit([&](sycl::handler & h) {
            sycl::local_accessor<int, 1> QS(sycl::range<1>(1024), h);
            sycl::local_accessor<sycl::float2, 1> DS(sycl::range<1>(128), h);
            constexpr int GN = T == T_IQ1_S ? 2048 : 512;
            sycl::local_accessor<uint32_t, 1> GS(sycl::range<1>(K3X_GRID_SLM ? GN : 1), h);
            h.parallel_for(sycl::nd_range<1>((nitems + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const long g0 = (long) it.get_group(0) * wg;
                const int e0 = ebase + (int) (g0 / ((long) nrows * nmat * (V7 ? 4 : 1)));
                const int lid = it.get_local_id(0);
                const int * qsrc = qt + (size_t) e0 * estride * 1024; const sycl::float2 * dsrc = dt + (size_t) e0 * estride * 128;
#if !V6
                if (xsrc) {   // fused q8_1 of the (shared) activation: block i -> SLM in the transposed layout, as store_T
                    for (int i = lid; i < xnb; i += wg) {
                        k3_q8_1 qb; k3_quantize_block(xsrc + (size_t) i * 32, &qb, mt_);
                        store_T(qb, i, QS.get_multi_ptr<sycl::access::decorated::no>().get(), DS.get_multi_ptr<sycl::access::decorated::no>().get());
                    }
                } else {
                    for (int i = lid; i < 1024; i += wg) QS[i] = qsrc[i];
                    for (int i = lid; i < 128; i += wg) DS[i] = dsrc[i];
                }
#if K3X_GRID_SLM
                { const uint32_t * gsrc = T == T_IQ1_S ? g1_ : (const uint32_t *) g2_; for (int i = lid; i < GN; i += wg) GS[i] = gsrc[i]; }
#endif
                sycl::group_barrier(it.get_group());
#endif
                const long gi = it.get_global_id(0);
                if (gi >= nitems) return;
#if V7
                // nitems here counts lanes: 4 per row
                const long sgi = gi / 16; const int lane = (int) (gi % 16);
                const int rg4 = (int) (sgi % (nrows / 4)), m = (int) ((sgi / (nrows / 4)) % nmat), e = ebase + (int) (sgi / ((long) (nrows / 4) * nmat));
                const uint32_t * mat = (const uint32_t *) (recs_.p[e] + moff + m * mstride);
#if K3X_GRID_SLM
                const uint32_t * gl = GS.get_multi_ptr<sycl::access::decorated::no>().get();
#else
                const uint32_t * gl = T == T_IQ1_S ? g1_ : (const uint32_t *) g2_;
#endif
                const float v7 = row_v7<T, NBLK>(it.get_sub_group(), mat, rg4, lane, QS.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                 DS.get_multi_ptr<sycl::access::decorated::no>().get(), gl, (const uint64_t *) gl);
                if (lane < 4) (m ? out1 : out0)[e * nrows + rg4 * 4 + lane] = v7;
#else
                const int r = (int) (gi % nrows), m = (int) ((gi / nrows) % nmat), e = ebase + (int) (gi / ((long) nrows * nmat));
                const uint32_t * mat = (const uint32_t *) (recs_.p[e] + moff + m * mstride);
#if V6
                const float v = row_v6<T, NBLK>(it.get_sub_group(), mat, r / 16, r % 16, qv + (size_t) e * estride * 896, dv + (size_t) e * estride * 224, g1_, g2_);
#else
                const float v = row_v5<T, NBLK>(mat, r / 16, r % 16, QS.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                 DS.get_multi_ptr<sycl::access::decorated::no>().get(), g1_, g2_);
#endif
                (m ? out1 : out0)[e * nrows + r] = v;
#endif
            });
        });
    }
    sycl::event matvec5(int k, int nmat, int nrows, int nblk, int t, size_t moff, size_t mstride,
                        const int * qt, const sycl::float2 * dt, int estride, float * out0, float * out1, const int * qv = nullptr, const float * dv = nullptr, int ebase = 0,
                        const float * xsrc = nullptr, int xnb = 0) {
        if (t == T_IQ1_S && nblk == 14) return matvec5_t<T_IQ1_S, 14>(k, nmat, nrows, moff, mstride, qt, dt, estride, out0, out1, qv, dv, ebase, xsrc, xnb);
        if (t == T_IQ1_S && nblk == 12) return matvec5_t<T_IQ1_S, 12>(k, nmat, nrows, moff, mstride, qt, dt, estride, out0, out1, qv, dv, ebase, xsrc, xnb);
        if (t == T_IQ2_XXS && nblk == 14) return matvec5_t<T_IQ2_XXS, 14>(k, nmat, nrows, moff, mstride, qt, dt, estride, out0, out1, qv, dv, ebase, xsrc, xnb);
        return matvec5_t<T_IQ2_XXS, 12>(k, nmat, nrows, moff, mstride, qt, dt, estride, out0, out1, qv, dv, ebase, xsrc, xnb);
    }

    template <int T, int NBLK>
    sycl::event matvec4_t(int k, int nmat, int nrows, size_t rb, size_t moff, size_t mstride,
                          const int * qv, const float * dv, int estride, float * out0, float * out1) {
        const uint32_t * g1_ = g1; const uint64_t * g2_ = g2;
        RecPtrs recs_; for (int i = 0; i < 16; i++) recs_.p[i] = recs[i];
        const long nitems = (long) k * nmat * nrows;   // multiple of 16 (rows are 3072 / 3584)
        const int wg = MV4_WG;
        constexpr int GN = T == T_IQ1_S ? 2048 : 512;
        return q.submit([&](sycl::handler & h) {
            sycl::local_accessor<uint32_t, 1> G(sycl::range<1>(GN), h);
            h.parallel_for(sycl::nd_range<1>((nitems + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid = it.get_local_id(0);
                const uint32_t * gsrc = T == T_IQ1_S ? g1_ : (const uint32_t *) g2_;
                for (int i = lid; i < GN; i += wg) G[i] = gsrc[i];
                sycl::group_barrier(it.get_group());
                const long gi = it.get_global_id(0);
                if (gi >= nitems) return;   // whole sub-groups (nitems % 16 == 0)
                const int r = (int) (gi % nrows), m = (int) ((gi / nrows) % nmat), e = (int) (gi / ((long) nrows * nmat));
                const uint8_t * rowp = recs_.p[e] + moff + m * mstride + (size_t) r * rb;
                const uint32_t * gl = G.get_multi_ptr<sycl::access::decorated::no>().get();
                const float v = row_v4<T, NBLK>(it.get_sub_group(), rowp, qv + (size_t) e * estride * 896, dv + (size_t) e * estride * 224, gl, (const uint64_t *) gl);
                (m ? out1 : out0)[e * nrows + r] = v;
            });
        });
    }
    sycl::event matvec4(int k, int nmat, int nrows, int nblk, int t, size_t rb, size_t moff, size_t mstride,
                        const int * qv, const float * dv, int estride, float * out0, float * out1) {
        if (t == T_IQ1_S && nblk == 14) return matvec4_t<T_IQ1_S, 14>(k, nmat, nrows, rb, moff, mstride, qv, dv, estride, out0, out1);
        if (t == T_IQ1_S && nblk == 12) return matvec4_t<T_IQ1_S, 12>(k, nmat, nrows, rb, moff, mstride, qv, dv, estride, out0, out1);
        if (t == T_IQ2_XXS && nblk == 14) return matvec4_t<T_IQ2_XXS, 14>(k, nmat, nrows, rb, moff, mstride, qv, dv, estride, out0, out1);
        return matvec4_t<T_IQ2_XXS, 12>(k, nmat, nrows, rb, moff, mstride, qv, dv, estride, out0, out1);
    }

    // v3: one row per work-item (16 consecutive rows per sub-group), see row_v3.
    template <int T, int NBLK>
    sycl::event matvec3_t(int k, int nmat, int nrows, size_t rb, size_t moff, size_t mstride,
                          const int * qt, const sycl::float2 * dt, int estride, float * out0, float * out1) {
        const uint32_t * g1_ = g1; const uint64_t * g2_ = g2;
        RecPtrs recs_; for (int i = 0; i < 16; i++) recs_.p[i] = recs[i];
        const long nitems = (long) k * nmat * nrows;
        const int wg = MV3_WG;
        return q.parallel_for(sycl::nd_range<1>((nitems + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const long gi = it.get_global_id(0);
            if (gi >= nitems) return;
            const int r = (int) (gi % nrows), m = (int) ((gi / nrows) % nmat), e = (int) (gi / ((long) nrows * nmat));
            const uint8_t * rowp = recs_.p[e] + moff + m * mstride + (size_t) r * rb;
            const float v = row_v3<T, NBLK>(rowp, qt + (size_t) e * estride * 1024, dt + (size_t) e * estride * 128, g1_, g2_);
            (m ? out1 : out0)[e * nrows + r] = v;
        });
    }
    sycl::event matvec3(int k, int nmat, int nrows, int nblk, int t, size_t rb, size_t moff, size_t mstride,
                        const int * qt, const sycl::float2 * dt, int estride, float * out0, float * out1) {
        if (t == T_IQ1_S && nblk == 14) return matvec3_t<T_IQ1_S, 14>(k, nmat, nrows, rb, moff, mstride, qt, dt, estride, out0, out1);
        if (t == T_IQ1_S && nblk == 12) return matvec3_t<T_IQ1_S, 12>(k, nmat, nrows, rb, moff, mstride, qt, dt, estride, out0, out1);
        if (t == T_IQ2_XXS && nblk == 14) return matvec3_t<T_IQ2_XXS, 14>(k, nmat, nrows, rb, moff, mstride, qt, dt, estride, out0, out1);
        return matvec3_t<T_IQ2_XXS, 12>(k, nmat, nrows, rb, moff, mstride, qt, dt, estride, out0, out1);
    }

    // v2: one sub-group (16 lanes) per row, RPS rows per sub-group in flight, no work-group barriers.
    // The row is staged through a per-sub-group SLM slice with coalesced dword loads; lane b decodes 256-block b
    // (CUDA lanes 8b..8b+7) and the shuffle reduction keeps the CUDA order. Rows map to sub-groups freely.
    sycl::event matvec2(int k, int nmat, int nrows, int nblk, int t, size_t rb, size_t moff, size_t mstride,
                        const int * qt, const sycl::float2 * dt, int estride, float * out0, float * out1) {
        const uint32_t * g1_ = g1; const uint64_t * g2_ = g2;
        RecPtrs recs_; for (int i = 0; i < 16; i++) recs_.p[i] = recs[i];
        constexpr int SGW = MV2_SG;                       // sub-groups per work-group
        const int ndw = (int) (rb / 4);                   // row bytes are a multiple of 4 (50*14, 66*14, 50*12, 66*12)
        const int slice = (ndw + 3) & ~3;
        const long total = (long) k * nmat * nrows;       // rows
        const long tasks = (total + MV2_RPS - 1) / MV2_RPS;
        const long nwgs = (tasks + SGW - 1) / SGW;
        return q.submit([&](sycl::handler & h) {
            sycl::local_accessor<uint32_t, 1> S(sycl::range<1>(SGW * MV2_RPS * slice), h);
            h.parallel_for(sycl::nd_range<1>(nwgs * SGW * 16, SGW * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int b = sg.get_local_linear_id(), sgi = sg.get_group_linear_id();
                const long task = (long) it.get_group(0) * SGW + sgi;
                uint32_t * my = &S[sgi * MV2_RPS * slice];
                // all loads of all RPS rows are issued before any SLM store, so their DRAM latencies overlap
                constexpr int MAXL = 15;                  // ceil(231 dwords / 16 lanes), the IQ2_XXS gate/up row
                uint32_t rg[MV2_RPS][MAXL];
#pragma unroll
                for (int j = 0; j < MV2_RPS; j++) {
                    const long row = std::min(task * MV2_RPS + j, total - 1);
                    const int r = (int) (row % nrows), m = (int) ((row / nrows) % nmat), e = (int) (row / ((long) nrows * nmat));
                    const uint32_t * src = (const uint32_t *) (recs_.p[e] + moff + m * mstride + (size_t) r * rb);
#pragma unroll
                    for (int u = 0; u < MAXL; u++) { const int i = b + 16 * u; if (i < ndw) rg[j][u] = src[i]; }
                }
#pragma unroll
                for (int j = 0; j < MV2_RPS; j++)
#pragma unroll
                    for (int u = 0; u < MAXL; u++) { const int i = b + 16 * u; if (i < ndw) my[j * slice + i] = rg[j][u]; }
                sycl::group_barrier(sg);
#pragma unroll
                for (int j = 0; j < MV2_RPS; j++) {
                    const long row = task * MV2_RPS + j;
                    if (row >= total) break;       // uniform across the sub-group
                    const int r = (int) (row % nrows), m = (int) ((row / nrows) % nmat), e = (int) (row / ((long) nrows * nmat));
                    float o[8];
                    if (b < nblk) {
                        const uint8_t * bp = (const uint8_t *) &my[j * slice] + b * k3_blk_bytes(t);
#if EXP == 2
                        for (int l = 0; l < 8; l++) o[l] = (float) bp[l];
                        if (0)
#endif
                        if (t == T_IQ1_S) blkT_iq1_s(bp, qt + (size_t) e * estride * 1024, dt + (size_t) e * estride * 128, b, g1_, o);
                        else blkT_iq2_xxs(bp, qt + (size_t) e * estride * 1024, dt + (size_t) e * estride * 128, b, g2_, o);
                    }
                    else for (int l = 0; l < 8; l++) o[l] = 0.0f;
                    float v[8];
#pragma unroll
                    for (int l = 0; l < 8; l++) {
                        const float a1 = sycl::select_from_group(sg, o[l], (b + 4) & 15);
                        const float a2 = sycl::select_from_group(sg, o[l], (b + 8) & 15);
                        const float a3 = sycl::select_from_group(sg, o[l], (b + 12) & 15);
                        v[l] = ((o[l] + a1) + a2) + a3;
                    }
#pragma unroll
                    for (int l = 0; l < 8; l++) v[l] = v[l] + sycl::select_from_group(sg, v[l], (b + 2) & 15);
#pragma unroll
                    for (int l = 0; l < 8; l++) v[l] = v[l] + sycl::select_from_group(sg, v[l], (b + 1) & 15);
#pragma unroll
                    for (int l = 0; l < 4; l++) v[l] = v[l] + v[l + 4];
                    v[0] = v[0] + v[2]; v[1] = v[1] + v[3];
                    v[0] = v[0] + v[1];
                    if (b == 0) (m ? out1 : out0)[e * nrows + r] = v[0];
                }
            });
        });
    }

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
        int * QT2_ = QT2; sycl::float2 * DT2_ = DT2; int * QV2_ = QV2; float * DV2_ = DV2;
        if (!ext_recs) for (int i = 0; i < k; i++) recs[i] = W + (size_t) ids[i] * prec;
        q.parallel_for(sycl::range<1>((size_t) k * F / 32), [=](sycl::id<1> i) {
            k3_quantize_block(hb + i[0] * 32, yq2_ + i[0], mt_);
            dsf2_[i[0]] = sycl::float2(k3_h2f(yq2_[i[0]].d), k3_h2f(yq2_[i[0]].s));
            store_T(yq2_[i[0]], (int) (i[0] % (F / 32)), QT2_ + (i[0] / (F / 32)) * 1024, DT2_ + (i[0] / (F / 32)) * 128);
            store_V(yq2_[i[0]], (int) (i[0] % (F / 32)), QV2_ + (i[0] / (F / 32)) * 896, DV2_ + (i[0] / (F / 32)) * 224);
        });
        if (V5) matvec5(k, 1, D, F / K3_QK, td, 2 * pg, 0, QT2, DT2, 1, rowsd, nullptr, QV2, DV2);
        else if (MV4) matvec4(k, 1, D, F / K3_QK, td, rbd, 2 * gsz, 0, QV2, DV2, 1, rowsd, nullptr);
        else if (MV3) matvec3(k, 1, D, F / K3_QK, td, rbd, 2 * gsz, 0, QT2, DT2, 1, rowsd, nullptr);
        else if (MV2) matvec2(k, 1, D, F / K3_QK, td, rbd, 2 * gsz, 0, QT2, DT2, 1, rowsd, nullptr);
        else matvec(k, 1, D, F / K3_QK, td, rbd, 2 * gsz, 0, 0, yq2, dsf2, F / 32, rowsd, nullptr);
        q.memcpy(rows, rowsd, (size_t) k * D * 4);
        q.wait_and_throw();
    }

    // one token: ids[0..k), x[D] -> rows[k][D]
    // one token: ids[0..k), x[D] -> rows[k][D]. With a sink, the down matmul runs per expert and sink(e) is called as
    // soon as row e is in host memory (in order), so the server can stream rows (C's engine takes partial replies).
    void run(int k, const std::function<void(int)> * sink = nullptr) {
        sycl::event tev[TR_NC]; bool tset[TR_NC] = {};
        auto T = [&](int i, const sycl::event & e) { if (tr && i < TR_NC) { tr->t_sub[i] = k3x_ns(); tev[i] = e; tset[i] = true; } };
        T(0, q.memcpy(xd, x, D * 4));
        const float * x_ = xd; float * rows_ = rowsd;
        if (!ext_recs) for (int i = 0; i < k; i++) recs[i] = W + (size_t) ids[i] * prec;
        k3_q8_1 * yq_ = yq, * yq2_ = yq2;
        int * QT_ = QT, * QT2_ = QT2; sycl::float2 * DT_ = DT, * DT2_ = DT2;
        float * gb = gbuf, * ub = ubuf, * hb = hbuf; const k3_mufu_tabs mt_ = mt;
        ev.clear();
#if K3X_FUSE_QX
        ev.push_back(matvec5(k, 2, F, D / K3_QK, tg, 0, pg, QT_, DT_, 0, gb, ub, nullptr, nullptr, 0, x_, D / 32));
        ev.push_back(ev.back());   // keeps the profiling slots: [0] = [1] = fused quant + gate/up
        T(2, ev.back());
#else
        ev.push_back(q.parallel_for(sycl::range<1>(D / 32), [=](sycl::id<1> i) {
            k3_quantize_block(x_ + i[0] * 32, yq_ + i[0], mt_);
            store_T(yq_[i[0]], (int) i[0], QT_, DT_);
        }));
        T(1, ev.back());
        ev.push_back(matvec5(k, 2, F, D / K3_QK, tg, 0, pg, QT_, DT_, 0, gb, ub));
        T(2, ev.back());
#endif
        // SiTU + q8_1 of h: 32 work-items per q8 block (one element each, latency-bound table lookups in parallel),
        // then one work-item quantizes the block from SLM exactly as k3_quantize_block does.
        ev.push_back(q.submit([&](sycl::handler & h) {
            sycl::local_accessor<float, 1> HS(sycl::range<1>(32), h);
            h.parallel_for(sycl::nd_range<1>((size_t) k * F, 32), [=](sycl::nd_item<1> it) {
                const size_t i = it.get_global_id(0); const int lid = it.get_local_id(0);
                const float hv = k3_act(gb[i], ub[i], mt_);
                hb[i] = hv; HS[lid] = hv;
                sycl::group_barrier(it.get_group());
                if (lid == 0) {
                    const size_t blk = i / 32; float a[32];
                    for (int j = 0; j < 32; j++) a[j] = HS[j];
                    k3_quantize_block(a, yq2_ + blk, mt_);
                    store_T(yq2_[blk], (int) (blk % (F / 32)), QT2_ + (blk / (F / 32)) * 1024, DT2_ + (blk / (F / 32)) * 128);
                }
            });
        }));
        T(3, ev.back());
        if (!sink) {
            ev.push_back(matvec5(k, 1, D, F / K3_QK, td, 2 * pg, 0, QT2_, DT2_, 1, rows_, nullptr));
            T(4, ev.back());
            ev.push_back(q.memcpy(rows, rowsd, (size_t) k * D * 4));
            T(5, ev.back());
            if (spin) { auto & last = ev.back(); while (last.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {} }
            if (tr) tr->t_row[0] = k3x_ns();
            q.wait_and_throw();
        } else {
            sycl::event cp[16];
            for (int e = 0; e < k; e++) {
                T(4 + 2 * e, matvec5(1, 1, D, F / K3_QK, td, 2 * pg, 0, QT2_, DT2_, 1, rows_, nullptr, nullptr, nullptr, e));
                cp[e] = q.memcpy(rows + (size_t) e * D, rowsd + (size_t) e * D, D * 4);
                T(5 + 2 * e, cp[e]);
            }
            for (int e = 0; e < k; e++) {
                while (cp[e].get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {}
                if (tr && e < TR_K) tr->t_row[e] = k3x_ns();
                (*sink)(e);
                if (tr && e < TR_K) tr->t_sent[e] = k3x_ns();
            }
            q.wait_and_throw();
        }
        if (tr) {
            tr->t_end = k3x_ns();
            if (g_tr_prof && sink)   // streamed: the reply went out already, so this is not on the request path
                for (int i = 0; i < TR_NC; i++) if (tset[i]) {
                    tr->d_sub[i]   = tev[i].get_profiling_info<sycl::info::event_profiling::command_submit>();
                    tr->d_start[i] = tev[i].get_profiling_info<sycl::info::event_profiling::command_start>();
                    tr->d_end[i]   = tev[i].get_profiling_info<sycl::info::event_profiling::command_end>();
                }
        }
        if (prof && ev.size() >= 4) {
            kspan += (ev.back().get_profiling_info<sycl::info::event_profiling::command_end>() -
                      ev.front().get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
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
    std::vector<char> v(n); if (fread(v.data(), 1, n, f) != (size_t) n) exit(1);
    posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);   // do not leave the file in the page cache (see serve_index)
    fclose(f); return v;
}
static k3_mufu_tabs host_mt{};
struct HostPacked { std::vector<uint32_t> hdr; std::vector<int8_t> res; std::vector<uint32_t> esc; uint32_t n = 0; };
static HostPacked tab_host[4];
static k3_mufu_tabs dev_mt_cache{}; static bool dev_mt_loaded = false;
static bool g_host_tables = true;   // keep the (packed) host copies for the host reference
// pack one raw MUFU table (u32 per entry) into blocks of 64: {a, d} + int8 residuals + escapes; verified exhaustively
static void pack_table(const uint32_t * t, uint32_t n, HostPacked & P) {
    const uint32_t nb = (n + 63) / 64;
    P.n = n; P.hdr.assign(2 * (size_t) nb, 0); P.res.assign(n, 0); P.esc.clear();
    for (uint32_t b = 0; b < nb; b++) {
        const uint32_t i0 = b * 64, len = std::min(64u, n - i0);
        const uint32_t a = t[i0];
        int64_t dd = len == 64 ? (int64_t) t[i0 + 63] - (int64_t) a : 0;
        if (dd > (1 << 24) || dd < -(1 << 24)) dd = 0;   // keeps d*o in int32; such blocks escape where needed
        const int32_t d = (int32_t) dd;
        P.hdr[2 * b] = a; P.hdr[2 * b + 1] = (uint32_t) d;
        for (uint32_t o = 0; o < len; o++) {
            const uint32_t ap = a + (uint32_t) ((d * (int32_t) o) / 63);
            const int64_t r = (int64_t) (int32_t) (t[i0 + o] - ap);
            if (r >= -127 && r <= 127) P.res[i0 + o] = (int8_t) r;
            else { P.res[i0 + o] = -128; P.esc.push_back(i0 + o); P.esc.push_back(t[i0 + o]); }
        }
    }
    k3_ptab v{P.hdr.data(), P.res.data(), P.esc.data(), (uint32_t) (P.esc.size() / 2), n};
    for (uint32_t i = 0; i < n; i++) if (k3_ptab_get(v, i) != t[i]) { fprintf(stderr, "pack verify FAILED at %u\n", i); exit(5); }
}
// loads ref/mufu/{rcp,ex2p,ex2n,tanh}.bin (K3X_MUFU=dir), packs them (~1.1 B/entry) and uploads the packed form
static void load_mufu(Engine & en) {
    if (dev_mt_loaded) { en.mt = dev_mt_cache; return; }
    const char * dir = getenv("K3X_MUFU") ? getenv("K3X_MUFU") : "../ref/mufu";
    const char * names[4] = {"rcp", "ex2p", "ex2n", "tanh"};
    k3_ptab dev[4] = {}; size_t devbytes = 0;
    for (int t = 0; t < 4; t++) {
        std::string p = std::string(dir) + "/" + names[t] + ".bin";
        FILE * f = fopen(p.c_str(), "rb");
        if (!f) { fprintf(stderr, "WARNING: no %s, IEEE placeholder\n", p.c_str()); continue; }
        fclose(f);
        HostPacked & P = tab_host[t];
        { auto raw = slurp(p); pack_table((const uint32_t *) raw.data(), (uint32_t) (raw.size() / 4), P); }
        uint32_t * h = sycl::malloc_device<uint32_t>(P.hdr.size(), en.q);
        int8_t * r = sycl::malloc_device<int8_t>(P.res.size(), en.q);
        uint32_t * e = sycl::malloc_device<uint32_t>(std::max<size_t>(P.esc.size(), 2), en.q);
        en.q.memcpy(h, P.hdr.data(), P.hdr.size() * 4); en.q.memcpy(r, P.res.data(), P.res.size());
        if (!P.esc.empty()) en.q.memcpy(e, P.esc.data(), P.esc.size() * 4);
        en.q.wait();
        { std::vector<int8_t> back(P.res.size()); en.q.memcpy(back.data(), r, back.size()).wait();
          if (memcmp(back.data(), P.res.data(), back.size())) { fprintf(stderr, "table upload verify FAILED %s\n", names[t]); exit(4); } }
        dev[t] = k3_ptab{h, r, e, (uint32_t) (P.esc.size() / 2), P.n};
        devbytes += P.hdr.size() * 4 + P.res.size() + P.esc.size() * 4;
        fprintf(stderr, "MUFU %s table: %u entries, packed %.0f MB, %zu escapes\n", names[t], P.n, (P.hdr.size() * 4 + P.res.size()) / 1e6, P.esc.size() / 2);
        if (!g_host_tables) { HostPacked().hdr.swap(P.hdr); std::vector<int8_t>().swap(P.res); std::vector<uint32_t>().swap(P.esc); }
    }
    en.mt = k3_mufu_tabs{dev[0], dev[1], dev[2], dev[3], dev[1].n, dev[2].n, dev[3].n};
    dev_mt_cache = en.mt; dev_mt_loaded = true;
    k3_ptab hp[4] = {};
    for (int t = 0; t < 4; t++) if (!tab_host[t].res.empty())
        hp[t] = k3_ptab{tab_host[t].hdr.data(), tab_host[t].res.data(), tab_host[t].esc.data(), (uint32_t) (tab_host[t].esc.size() / 2), tab_host[t].n};
    host_mt = k3_mufu_tabs{hp[0], hp[1], hp[2], hp[3], hp[1].n, hp[2].n, hp[3].n};
    fprintf(stderr, "MUFU tables on the GPU: %.2f GB\n", devbytes / 1e9);
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }


// first (K3X_TRACE): set to the time the first bytes arrived; a SIGUSR1 dump runs here while nothing has arrived yet
static bool rd_spin(int fd, void * p, size_t n, uint64_t * first = nullptr) {
    char * c = (char *) p;
    while (n) {
        ssize_t r = recv(fd, c, n, MSG_DONTWAIT);
        if (r == 0) return false;
        if (r < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) { if (g_tr_dump && first && !*first) k3x_tr_dump(); continue; }
            return false;
        }
        if (first && !*first) *first = k3x_ns();
        c += r; n -= r;
    }
    return true;
}
static bool wr_all(int fd, const void * p, size_t n) { const char * c = (const char *) p; while (n) { ssize_t r = send(fd, c, n, 0); if (r <= 0) return false; c += r; n -= r; } return true; }

// serve agent C's remote set: IDX lines "<global_id> <tg> <td> <byte_offset>", BIN = records back to back.
// Weights go to device USM in chunks of <= 1 GiB (no >4 GiB buffers needed); one Engine (scratch + kernels) per (tg, td).
static int serve_index(const char * idxp, const char * binp, int port) {
    g_host_tables = getenv("K3X_FULLCHECK") != nullptr;
    const bool stream = !getenv("K3X_NOSTREAM");
    struct R { long gid; int tg, td; size_t off; };
    std::vector<R> idx; { FILE * f = fopen(idxp, "r"); if (!f) { perror(idxp); return 1; } R r; while (fscanf(f, "%ld %d %d %zu", &r.gid, &r.tg, &r.td, &r.off) == 4) idx.push_back(r); fclose(f); }
    std::vector<Engine *> eng; auto get = [&](int tg, int td) -> Engine * { for (auto * e : eng) if (e->tg == tg && e->td == td) return e; eng.push_back(new Engine(tg, td, 0)); eng.back()->spin = true; load_mufu(*eng.back()); return eng.back(); };
    std::unordered_map<long, std::pair<Engine *, const uint8_t *>> map;
    FILE * f = fopen(binp, "rb"); if (!f) { perror(binp); return 1; }
    const size_t CH = (size_t) 1 << 30; uint8_t * chunk = nullptr; size_t used = CH; std::vector<uint8_t> buf; uint8_t * stage = nullptr; const size_t maxrec = (size_t) 16 << 20;
    double t0 = now_ms(); size_t tot = 0;
    // The Arc iGPU silently reads ZEROS from USM beyond roughly 25-35 GB mapped per process (measured 2026-10-28: 3500
    // records = 23 GB fine, 5000 = 33 GB and 6967 = 46 GB all-zero rows, device and host USM alike). Stop loading at
    // K3X_MAX_GB (default 24): ids past it are unknown and get NaN rows instead of silently wrong ones.
    const double max_gb = getenv("K3X_MAX_GB") ? atof(getenv("K3X_MAX_GB")) : 24.0;
    for (auto & r : idx) {
        Engine * e = get(r.tg, r.td);
        if ((tot + e->prec) / 1e9 > max_gb) { fprintf(stderr, "K3X_MAX_GB %.1f reached: serving %zu of %zu records\n", max_gb, map.size(), idx.size()); break; }
        if (used + e->prec > CH) { chunk = getenv("K3X_HOSTW") ? sycl::malloc_host<uint8_t>(CH, e->q) : sycl::malloc_device<uint8_t>(CH, e->q); if (!chunk) { fprintf(stderr, "device alloc failed after %.1f GB\n", tot / 1e9); return 1; } used = 0; }
        buf.resize(e->rec);
        if (fseeko(f, (off_t) r.off, SEEK_SET) || fread(buf.data(), 1, e->rec, f) != e->rec) { fprintf(stderr, "short read gid %ld\n", r.gid); return 1; }
        // Drop the record from the page cache right away: a page cache that grows with the 46 GB file squeezes memory
        // while ~48 GB of GPU objects are being populated, which is when the silent all-zero failures happen.
        if (!getenv("K3X_KEEPCACHE")) posix_fadvise(fileno(f), (off_t) r.off, (off_t) e->rec, POSIX_FADV_DONTNEED);
        // Stage through pinned USM host memory: copying from a pageable std::vector made the runtime import it as an
        // i915 userptr, and those submissions sometimes failed (EXECBUFFER2 EFAULT) without any error, leaving the
        // record zero on the GPU. Each record is then spot-checked by reading three 256-byte windows back.
        if (!stage) stage = sycl::malloc_host<uint8_t>(maxrec, e->q);
        e->repack(buf.data(), stage);
        e->q.memcpy(chunk + used, stage, e->prec).wait();
        for (size_t off : {(size_t) 0, e->prec / 2, e->prec - 256}) {
            uint8_t back[256]; e->q.memcpy(back, chunk + used + off, 256).wait();
            if (memcmp(back, stage + off, 256)) { fprintf(stderr, "upload verify FAILED for gid %ld at +%zu\n", r.gid, off); exit(4); }
        }
        map[r.gid] = {e, chunk + used}; used += e->prec; tot += e->prec;
    }
    fclose(f);
    if (!getenv("K3X_FULLCHECK")) for (auto & v : tab_host) v = HostPacked();   // host copies only serve the host reference
    fprintf(stderr, "loaded %zu of %zu records (%.1f GB on the GPU) in %.1f s, %zu engine kinds\n", map.size(), idx.size(), tot / 1e9, (now_ms() - t0) / 1e3, eng.size());
    if (const char * so = getenv("K3X_SETOUT")) {   // "layer expert" of the served records, for LLAMA_EXPERT_STORE_REMOTE_SET
        FILE * f = fopen(so, "w");
        for (auto & r : idx) if (map.count(r.gid)) fprintf(f, "%ld %ld\n", r.gid / 896, r.gid % 896);
        fclose(f);
    }
    if (getenv("K3X_ROWSOUT") || getenv("K3X_ROWSREF")) {
        // K3X_ROWSOUT: append (gid, row of one fixed token) for every served record; K3X_ROWSREF: compare every served
        // record with such a file (built from small known-good subsets) bit for bit. Fixed token: same x as FULLCHECK.
        std::vector<float> xin(D); for (int i = 0; i < D; i++) xin[i] = 0.03f * (float) ((i * 53) % 67 - 33);
        std::unordered_map<long, std::vector<float>> ref;
        if (const char * rp = getenv("K3X_ROWSREF")) {
            FILE * f = fopen(rp, "rb"); if (!f) { perror(rp); exit(1); }
            int64_t g; std::vector<float> row(D);
            while (fread(&g, 8, 1, f) == 1 && fread(row.data(), 4, D, f) == (size_t) D) ref[(long) g] = row;
            fclose(f);
        }
        FILE * fo = getenv("K3X_ROWSOUT") ? fopen(getenv("K3X_ROWSOUT"), "ab") : nullptr;
        long n = 0, nref = 0, badrec = 0, badval = 0, zero = 0; double t1 = now_ms();
        for (auto & r : idx) {
            auto it = map.find(r.gid); if (it == map.end()) continue;
            Engine * e = it->second.first; e->ext_recs = true;
            memcpy(e->x, xin.data(), D * 4); e->recs[0] = it->second.second; e->run(1);
            int nz = 0; for (int i = 0; i < D; i++) nz += e->rows[i] != 0.0f; zero += nz == 0;
            if (fo) { int64_t g = r.gid; fwrite(&g, 8, 1, fo); fwrite(e->rows, 4, D, fo); }
            auto rf = ref.find(r.gid);
            if (rf != ref.end()) { nref++; long bv = 0; for (int i = 0; i < D; i++) bv += k3_f2u(rf->second[i]) != k3_f2u(e->rows[i]); if (bv) { badrec++; badval += bv; } }
            n++;
        }
        if (fo) fclose(fo);
        fprintf(stderr, "ROWCHECK: %ld records run in %.1f s, %ld all-zero; vs reference: %ld compared, %ld differ (%ld values)\n",
                n, (now_ms() - t1) / 1e3, zero, nref, badrec, badval);
        if (zero || badrec) { fprintf(stderr, "ROWCHECK FAILED\n"); exit(6); }
    }
    if (getenv("K3X_FULLCHECK")) {   // every served record: one token on the GPU vs the host reference, bit for bit
        std::vector<const R *> served; for (auto & r : idx) if (map.count(r.gid)) served.push_back(&r);
        const size_t ns = served.size();
        std::vector<float> xin(D); for (int i = 0; i < D; i++) xin[i] = 0.03f * (float) ((i * 53) % 67 - 33);
        std::vector<float> devrows(ns * D);
        double t1 = now_ms();
        for (size_t s2 = 0; s2 < ns; s2++) {
            auto it = map.find(served[s2]->gid); Engine * e = it->second.first; e->ext_recs = true;
            memcpy(e->x, xin.data(), D * 4); e->recs[0] = it->second.second; e->run(1);
            memcpy(&devrows[s2 * D], e->rows, D * 4);
        }
        fprintf(stderr, "FULLCHECK: GPU pass %.1f s, host reference on %u threads...\n", (now_ms() - t1) / 1e3, std::thread::hardware_concurrency());
        std::atomic<size_t> next{0}, badrec{0}; std::atomic<long> badval{0};
        auto worker = [&]() {
            FILE * f = fopen(binp, "rb"); std::vector<uint8_t> rb; std::vector<float> ref(D); int zero = 0;
            for (size_t s2; (s2 = next++) < ns;) {
                const R & r = *served[s2]; const size_t rec = map.find(r.gid)->second.first->rec;
                rb.resize(rec); fseeko(f, (off_t) r.off, SEEK_SET); if (fread(rb.data(), 1, rec, f) != rec) { badrec++; continue; }
                host_ref(rb.data(), rec, r.tg, r.td, &zero, 1, xin.data(), ref.data(), host_mt);
                long bv = 0; for (int i = 0; i < D; i++) bv += k3_f2u(ref[i]) != k3_f2u(devrows[s2 * D + i]);
                if (bv) { badrec++; badval += bv; }
            }
            fclose(f);
        };
        std::vector<std::thread> th; for (unsigned i = 0; i < std::max(1u, std::thread::hardware_concurrency() - 2); i++) th.emplace_back(worker);
        for (auto & t : th) t.join();
        fprintf(stderr, "FULLCHECK: %zu of %zu records bit-identical to the host reference (%ld values differ), %.1f s\n",
                ns - badrec.load(), ns, badval.load(), (now_ms() - t1) / 1e3);
        if (badrec) exit(6);
        for (auto & v : tab_host) v = HostPacked();
    }
    if (getenv("K3X_PROBEALL")) {   // diagnostic: one token through every served record, report all-zero rows by load order
        long bad = 0, first = -1, lastb = -1, n = 0; std::vector<int> badpos;
        for (auto & r : idx) {
            auto it = map.find(r.gid); if (it == map.end()) continue;
            Engine * e = it->second.first; e->ext_recs = true;
            for (int i = 0; i < D; i++) e->x[i] = 0.05f * (float) ((i * 37) % 41 - 20);
            e->recs[0] = it->second.second; e->run(1);
            int nz = 0; for (int i = 0; i < D; i++) nz += e->rows[i] != 0.0f;
            if (!nz) { bad++; if (first < 0) first = n; lastb = n; }
            n++;
        }
        fprintf(stderr, "PROBEALL: %ld of %ld records gave all-zero rows (first at load position %ld, last %ld)\n", bad, n, first, lastb);
    }
    // warm-up + self-check: a real record cannot give an all-zero row. All-zero rows were seen when a big server was
    // started right after killing another one (the i915 driver frees the old process's pinned pages lazily, so the
    // new pages could not all be made resident and the kernels silently did nothing): refuse to serve then.
    for (auto * e : eng) {
        e->ext_recs = true; for (int i = 0; i < D; i++) e->x[i] = 0.05f * (float) ((i * 37) % 41 - 20);
        const uint8_t * last = nullptr; for (auto & kv : map) if (kv.second.first == e) { if (!last || kv.second.second > last) last = kv.second.second; }
        for (const uint8_t * probe : {map.begin()->second.second, last}) {
            if (!probe) continue;
            if (map.begin()->second.first != e && probe == map.begin()->second.second) continue;
            e->recs[0] = probe; e->run(1);
            int nz = 0; for (int i = 0; i < D; i++) nz += e->rows[i] != 0.0f;
            if (nz == 0) { fprintf(stderr, "SELF-CHECK FAILED: all-zero row (GPU memory not resident?); check MemAvailable / dmesg, retry later\n"); exit(3); }
        }
    }
    fprintf(stderr, "self-check ok\n");
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(ls, (sockaddr *) &a, sizeof a) || listen(ls, 4)) { perror("bind"); return 1; }
    fprintf(stderr, "serving on %d (busy-poll)\n", port);
    k3x_tr_init();
    std::vector<float> nanrows(16 * D, NAN), xin(D); long zero_rows = 0; int nconn = 0;
    for (;;) {
        int c = accept(ls, nullptr, nullptr);
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        int32_t hdr[17]; nconn++;
        for (;;) {
            k3x_tr * tr = k3x_tr_begin();
            uint64_t t_first = 0;
            if (!rd_spin(c, hdr, 17 * 4, tr ? &t_first : nullptr)) break;
            if (tr) { tr->t_first = t_first; tr->t_hdr = k3x_ns(); }
            if (!rd_spin(c, xin.data(), D * 4)) break;
            const int k = hdr[0];
            if (tr) { tr->t_rx = k3x_ns(); tr->k = k; tr->conn = nconn; tr->seq = k >= 1 && k < 16 ? hdr[16] : -1; tr->f_before = k3x_tr_freq(); }
            if (k < 1 || k > 16) break;
            auto it0 = map.find(hdr[1]);
            Engine * e = it0 == map.end() ? nullptr : it0->second.first;
            bool ok = e != nullptr;
            for (int i = 0; ok && i < k; i++) { auto it = map.find(hdr[1 + i]); if (it == map.end() || it->second.first != e) ok = false; else e->recs[i] = it->second.second; }
            if (!ok) { if (!wr_all(c, nanrows.data(), (size_t) k * D * 4)) break; if (tr) tr->t_done = k3x_ns(); continue; }
            memcpy(e->x, xin.data(), D * 4);
            if (tr) tr->t_x = k3x_ns();
            e->tr = tr;
            bool okw = true;
            // A dropped GPU submission (i915 EXECBUFFER2 EFAULT under a very large residency set) leaves an all-zero
            // row; a real expert row is never all zero, so such a row is sent as NaN and the client reads it locally.
            auto guard = [&](int j) { const float * r = e->rows + (size_t) j * D; for (int i = 0; i < D; i++) if (r[i] != 0.0f) return r;
                                      zero_rows++; return (const float *) nanrows.data(); };
            if (stream) { std::function<void(int)> snk = [&](int j) { if (okw) okw = wr_all(c, guard(j), D * 4); }; e->run(k, &snk); }
            else { e->run(k); for (int j = 0; j < k && okw; j++) okw = wr_all(c, guard(j), D * 4); if (tr) tr->t_sent[0] = k3x_ns(); }
            e->tr = nullptr;
            if (tr) { tr->t_done = k3x_ns(); tr->f_after = k3x_tr_freq(); }
            if (!okw) break;
        }
        close(c);
        k3x_tr_dump();
    }
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: k3x RECORDS N TG TD check|bench|serve ...\n"); return 1; }
    std::string path = argv[1]; int E = atoi(argv[2]), tg = atoi(argv[3]), td = atoi(argv[4]); std::string mode = argv[5];
    if (mode == "serveidx") return serve_index(argv[1], argv[2], atoi(argv[6]));
    Engine en(tg, td, E);
    std::vector<uint8_t> host;
    if (getenv("K3X_SPIN")) en.spin = true;
    g_host_tables = mode != "serve";
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
    if (mode == "mufustat") {   // residuals of block-linear interpolation of the MUFU tables (bit space), per block size
        const char * dir = getenv("K3X_MUFU") ? getenv("K3X_MUFU") : "../ref/mufu";
        for (const char * nm : {"rcp", "ex2p", "ex2n", "tanh"}) {
            auto v = slurp(std::string(dir) + "/" + nm + ".bin"); const uint32_t * t = (const uint32_t *) v.data(); const size_t n = v.size() / 4;
            for (int B : {16, 32, 64}) {
                long over8 = 0, over16 = 0; long maxr = 0;
                for (size_t b0 = 0; b0 < n; b0 += B) {
                    const size_t len = std::min((size_t) B, n - b0); const int64_t a = t[b0], z = t[b0 + len - 1];
                    for (size_t i = 0; i < len; i++) {
                        const int64_t ap = len > 1 ? a + (z - a) * (int64_t) i / (int64_t) (len - 1) : a;
                        const int64_t r = (int64_t) t[b0 + i] - ap; const long ar = r < 0 ? -r : r;
                        if (ar > maxr) maxr = ar; if (r < -128 || r > 127) over8++; if (r < -32768 || r > 32767) over16++;
                    }
                }
                printf("%s B=%d: n=%zu max|r|=%ld  >int8: %ld  >int16: %ld\n", nm, B, n, maxr, over8, over16); fflush(stdout);
            }
        }
        return 0;
    }
    if (mode == "info") {
        auto d = en.q.get_device();
        printf("global_mem_size %.2f GB, max_mem_alloc_size %.2f GB, address_bits %u\n", d.get_info<sycl::info::device::global_mem_size>() / 1e9,
               d.get_info<sycl::info::device::max_mem_alloc_size>() / 1e9, d.get_info<sycl::info::device::address_bits>());
        if (d.has(sycl::aspect::ext_intel_free_memory)) printf("free_memory %.2f GB\n", d.get_info<sycl::ext::intel::info::device::free_memory>() / 1e9);
        return 0;
    }
    if (mode == "ballast") {   // allocate N GB of USM (argv[6], device|host argv[7]), touch it, then validate argv[8]
        const double gb = atof(argv[6]); const bool hostusm = std::string(argv[7]) == "host";
        std::vector<void *> v; size_t tot = 0;
        while (tot + ((size_t) 1 << 30) <= (size_t) (gb * 1e9)) {
            void * p = hostusm ? (void *) sycl::malloc_host<uint8_t>((size_t) 1 << 30, en.q) : (void *) sycl::malloc_device<uint8_t>((size_t) 1 << 30, en.q);
            if (!p) { printf("alloc failed at %.1f GB\n", tot / 1e9); break; }
            en.q.memset(p, 1, (size_t) 1 << 30).wait(); v.push_back(p); tot += (size_t) 1 << 30;
        }
        printf("ballast %.1f GB %s\n", tot / 1e9, hostusm ? "host" : "device");
        // quick correctness probe: one token on record 0 vs host reference
        for (int i = 0; i < D; i++) en.x[i] = nd(rng) * 0.5f;
        en.ids[0] = 0; en.run(1);
        std::vector<float> ref(D);
        host_ref(host.data(), en.rec, tg, td, en.ids, 1, en.x, ref.data(), host_mt);
        int same = 0, zero = 0; for (int i = 0; i < D; i++) { same += k3_f2u(ref[i]) == k3_f2u(en.rows[i]); zero += en.rows[i] == 0.0f; }
        printf("probe: %d/%d identical, %d zeros\n", same, D, zero);
        return 0;
    }
    if (mode == "membw") {   // ceiling: stream the record buffer with uint4 loads; and an empty kernel's cost
        sycl::queue & q = en.q; const size_t n16 = en.rec * E / 16; const sycl::uint4 * src = (const sycl::uint4 *) en.W;
        uint32_t * sink = sycl::malloc_device<uint32_t>(1, q);
        for (int wg : {64, 128, 256, 512}) for (int per : {1, 4, 16}) {
            const size_t nitems = n16 / per / wg * wg;
            std::vector<double> t;
            for (int it = 0; it < 8; it++) {
                double t0 = now_ms();
                q.parallel_for(sycl::nd_range<1>(nitems, wg), [=](sycl::nd_item<1> i) {
                    const size_t g = i.get_global_id(0); uint32_t acc = 0;
                    for (int j = 0; j < per; j++) { const sycl::uint4 v = src[g + (size_t) j * nitems]; acc ^= v.x() ^ v.y() ^ v.z() ^ v.w(); }
                    if (acc == 0x12345678u) sink[0] = acc;
                }).wait();
                t.push_back(now_ms() - t0);
            }
            std::sort(t.begin(), t.end());
            printf("membw wg %d per %d: %.3f ms -> %.1f GB/s\n", wg, per, t[0], nitems * per * 16 / t[0] / 1e6);
        }
        std::vector<double> t;
        for (int it = 0; it < 200; it++) { double t0 = now_ms(); q.single_task([=]() { sink[0] = 1; }).wait(); t.push_back(now_ms() - t0); }
        std::sort(t.begin(), t.end());
        printf("empty kernel + wait: p50 %.3f ms min %.3f ms\n", t[100], t[0]);
        t.clear();
        for (int it = 0; it < 200; it++) { double t0 = now_ms(); for (int j = 0; j < 5; j++) q.single_task([=]() { sink[0] = 1; }); q.wait(); t.push_back(now_ms() - t0); }
        std::sort(t.begin(), t.end());
        printf("5 empty kernels + wait: p50 %.3f ms\n", t[100]);
        return 0;
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
            std::vector<double> t, tfirst;
            for (int it = 0; it < iters + 5; it++) {
                for (int i = 0; i < k; i++) en.ids[i] = rng() % E;
                double t0 = now_ms(), tf = 0;
                if (getenv("K3X_STREAM")) { std::function<void(int)> snk = [&](int j) { if (j == 0) tf = now_ms() - t0; }; en.run(k, &snk); }
                else en.run(k);
                double dt = now_ms() - t0;
                if (it >= 5) { t.push_back(dt); tfirst.push_back(tf); }
            }
            std::sort(t.begin(), t.end());
            double p50 = t[t.size() / 2], p90 = t[(size_t) (t.size() * 0.9)];
            if (en.prof) { printf("  kernels (ms avg): quant %.3f gate/up %.3f situ+quant %.3f down %.3f  gpu span %.3f\n", en.kt[0] / en.nrun, en.kt[1] / en.nrun, en.kt[2] / en.nrun, en.kt[3] / en.nrun, en.kspan / en.nrun); for (double & v : en.kt) v = 0; en.kspan = 0; en.nrun = 0; }
            if (getenv("K3X_STREAM")) { std::sort(tfirst.begin(), tfirst.end()); printf("  first row p50 %.3f ms\n", tfirst[tfirst.size() / 2]); }
            printf("k=%2d: p50 %.3f ms  p90 %.3f ms  min %.3f ms  -> %.1f GB/s (p50)\n", k, p50, p90, t[0], k * en.rec / p50 / 1e6);
            fflush(stdout);
        }
        return 0;
    }
    if (mode == "serve") {
        const bool stream = !getenv("K3X_NOSTREAM");
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
                bool okw = true;
                if (stream) { std::function<void(int)> snk = [&](int j) { if (okw) okw = wr(c, en.rows + (size_t) j * D, D * 4); }; en.run(k, &snk); }
                else { en.run(k); okw = wr(c, en.rows, (size_t) k * D * 4); }
                if (!okw) break;
            }
            close(c);
        }
    }
    return 0;
}
