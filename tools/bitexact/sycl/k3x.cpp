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
#include <functional>
#include <utility>
#include <algorithm>
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
#if defined(__SYCL_DEVICE_ONLY__) && EXP == 5
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

template <int T, int NBLK>
static inline float row_v7(sycl::sub_group sg, const uint32_t * mat, int rg4, int lane, const int * QT, const sycl::float2 * DT, const uint32_t * g1, const uint64_t * g2) {
    constexpr int NDW = T == T_IQ1_S ? 13 : 17, NW = (T == T_IQ1_S ? 50 : 66) / 2;
    const int q = lane / 4;   // kb0
    float s[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int kb = q + 4 * i;
        const uint32_t * bp = mat + (((size_t) rg4 * 4 + i) * NDW) * 16 + lane;
        uint32_t d[NDW];
#pragma unroll
        for (int u = 0; u < NDW; u++) d[u] = bp[u * 16];
        if (kb < NBLK) {
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
#define MV5_WG 256
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
static inline size_t v7_mat_bytes(int t, int nrows) { return (size_t) (nrows / 4) * 4 * v5_ndw(t) * 64; }
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
                        uint8_t b4[4] = {0, 0, 0, 0};
                        if (kb < nblk) for (int c = 0; c < 4; c++) if (u * 4 + c < bb) b4[c] = src[(size_t) (rg * 4 + j) * rb + (size_t) kb * bb + u * 4 + c];
                        uint32_t v; memcpy(&v, b4, 4);
                        o[(((size_t) rg * 4 + i) * ndw + u) * 16 + q * 4 + j] = v;
                    }
}
struct RecPtrs { const uint8_t * p[16]; };
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
        pg = V7 ? v7_mat_bytes(tg, F) : v5_mat_bytes(tg, F, D); pd = V7 ? v7_mat_bytes(td, D) : v5_mat_bytes(td, D, F); prec = V5 ? 2 * pg + pd : rec;
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
        std::vector<uint8_t> tmp(prec);
        for (size_t i = 0; i < n; i++) { repack(host + i * rec, tmp.data()); q.memcpy(W + (e0 + i) * prec, tmp.data(), prec).wait(); }
    }

    template <int T, int NBLK>
    sycl::event matvec5_t(int k, int nmat, int nrows, size_t moff, size_t mstride,
                          const int * qt, const sycl::float2 * dt, int estride, float * out0, float * out1,
                          const int * qv = nullptr, const float * dv = nullptr, int ebase = 0) {
        const uint32_t * g1_ = g1; const uint64_t * g2_ = g2;
        RecPtrs recs_; for (int i = 0; i < 16; i++) recs_.p[i] = recs[i];
        const long nitems = (long) k * nmat * nrows * (V7 ? 4 : 1);
        const int wg = MV5_WG;   // divides 3072 and 3584, so a work-group never spans two (expert, matrix) pairs
        return q.submit([&](sycl::handler & h) {
            sycl::local_accessor<int, 1> QS(sycl::range<1>(1024), h);
            sycl::local_accessor<sycl::float2, 1> DS(sycl::range<1>(128), h);
            h.parallel_for(sycl::nd_range<1>((nitems + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const long g0 = (long) it.get_group(0) * wg;
                const int e0 = ebase + (int) (g0 / ((long) nrows * nmat * (V7 ? 4 : 1)));
                const int lid = it.get_local_id(0);
                const int * qsrc = qt + (size_t) e0 * estride * 1024; const sycl::float2 * dsrc = dt + (size_t) e0 * estride * 128;
#if !V6
                for (int i = lid; i < 1024; i += wg) QS[i] = qsrc[i];
                for (int i = lid; i < 128; i += wg) DS[i] = dsrc[i];
                sycl::group_barrier(it.get_group());
#endif
                const long gi = it.get_global_id(0);
                if (gi >= nitems) return;
#if V7
                // nitems here counts lanes: 4 per row
                const long sgi = gi / 16; const int lane = (int) (gi % 16);
                const int rg4 = (int) (sgi % (nrows / 4)), m = (int) ((sgi / (nrows / 4)) % nmat), e = ebase + (int) (sgi / ((long) (nrows / 4) * nmat));
                const uint32_t * mat = (const uint32_t *) (recs_.p[e] + moff + m * mstride);
                const float v7 = row_v7<T, NBLK>(it.get_sub_group(), mat, rg4, lane, QS.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                 DS.get_multi_ptr<sycl::access::decorated::no>().get(), g1_, g2_);
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
                        const int * qt, const sycl::float2 * dt, int estride, float * out0, float * out1, const int * qv = nullptr, const float * dv = nullptr, int ebase = 0) {
        if (t == T_IQ1_S && nblk == 14) return matvec5_t<T_IQ1_S, 14>(k, nmat, nrows, moff, mstride, qt, dt, estride, out0, out1, qv, dv, ebase);
        if (t == T_IQ1_S && nblk == 12) return matvec5_t<T_IQ1_S, 12>(k, nmat, nrows, moff, mstride, qt, dt, estride, out0, out1, qv, dv, ebase);
        if (t == T_IQ2_XXS && nblk == 14) return matvec5_t<T_IQ2_XXS, 14>(k, nmat, nrows, moff, mstride, qt, dt, estride, out0, out1, qv, dv, ebase);
        return matvec5_t<T_IQ2_XXS, 12>(k, nmat, nrows, moff, mstride, qt, dt, estride, out0, out1, qv, dv, ebase);
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
        q.memcpy(xd, x, D * 4);
        const float * x_ = xd; float * rows_ = rowsd;
        if (!ext_recs) for (int i = 0; i < k; i++) recs[i] = W + (size_t) ids[i] * prec;
        k3_q8_1 * yq_ = yq, * yq2_ = yq2;
        int * QT_ = QT, * QT2_ = QT2; sycl::float2 * DT_ = DT, * DT2_ = DT2;
        float * gb = gbuf, * ub = ubuf, * hb = hbuf; const k3_mufu_tabs mt_ = mt;
        ev.clear();
        ev.push_back(q.parallel_for(sycl::range<1>(D / 32), [=](sycl::id<1> i) {
            k3_quantize_block(x_ + i[0] * 32, yq_ + i[0], mt_);
            store_T(yq_[i[0]], (int) i[0], QT_, DT_);
        }));
        ev.push_back(matvec5(k, 2, F, D / K3_QK, tg, 0, pg, QT_, DT_, 0, gb, ub));
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
        if (!sink) {
            ev.push_back(matvec5(k, 1, D, F / K3_QK, td, 2 * pg, 0, QT2_, DT2_, 1, rows_, nullptr));
            ev.push_back(q.memcpy(rows, rowsd, (size_t) k * D * 4));
            if (spin) { auto & last = ev.back(); while (last.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {} }
            q.wait_and_throw();
        } else {
            sycl::event cp[16];
            for (int e = 0; e < k; e++) {
                matvec5(1, 1, D, F / K3_QK, td, 2 * pg, 0, QT2_, DT2_, 1, rows_, nullptr, nullptr, nullptr, e);
                cp[e] = q.memcpy(rows + (size_t) e * D, rowsd + (size_t) e * D, D * 4);
            }
            for (int e = 0; e < k; e++) {
                while (cp[e].get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {}
                (*sink)(e);
            }
            q.wait_and_throw();
        }
        if (prof && ev.size() >= 4) {
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
static bool g_host_tables = true;   // serve modes need no host copies (only the host reference uses them)
static void load_mufu(Engine & en) {
    if (dev_mt_loaded) { en.mt = dev_mt_cache; return; }
    const char * dir = getenv("K3X_MUFU") ? getenv("K3X_MUFU") : "../ref/mufu";
    const char * names[4] = {"rcp", "ex2p", "ex2n", "tanh"};
    const uint32_t * dev[4] = {nullptr, nullptr, nullptr, nullptr}; uint32_t n[4] = {0, 0, 0, 0};
    for (int t = 0; t < 4; t++) {
        std::string p = std::string(dir) + "/" + names[t] + ".bin";
        FILE * f = fopen(p.c_str(), "rb");
        if (!f) { fprintf(stderr, "WARNING: no %s, IEEE placeholder\n", p.c_str()); continue; }
        fseek(f, 0, SEEK_END); const size_t bytes = (size_t) ftell(f); fseek(f, 0, SEEK_SET);
        n[t] = (uint32_t) (bytes / 4);
        uint32_t * d = sycl::malloc_device<uint32_t>(n[t], en.q);
        if (g_host_tables) {
            fclose(f);
            if (tab_host[t].empty()) tab_host[t] = slurp(p);
            en.q.memcpy(d, tab_host[t].data(), bytes).wait();
        } else {   // stream through a 64 MiB buffer: no 2.5 GB host copy at startup
            std::vector<char> buf((size_t) 64 << 20);
            for (size_t off = 0; off < bytes;) {
                const size_t c = std::min(buf.size(), bytes - off);
                if (fread(buf.data(), 1, c, f) != c) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
                en.q.memcpy((char *) d + off, buf.data(), c).wait();
                off += c;
            }
            fclose(f);
        }
        dev[t] = d;
        fprintf(stderr, "MUFU %s table: %u entries\n", names[t], n[t]);
    }
    en.mt = k3_mufu_tabs{dev[0], dev[1], dev[2], dev[3], n[1], n[2], n[3]};
    dev_mt_cache = en.mt; dev_mt_loaded = true;
    host_mt = k3_mufu_tabs{(const uint32_t *) tab_host[0].data(), (const uint32_t *) tab_host[1].data(), (const uint32_t *) tab_host[2].data(),
                           (const uint32_t *) tab_host[3].data(), n[1], n[2], n[3]};
    for (int t = 0; t < 4; t++) if (tab_host[t].empty()) { if (t == 0) host_mt.rcp = nullptr; if (t == 1) host_mt.ex2p = nullptr; if (t == 2) host_mt.ex2n = nullptr; if (t == 3) host_mt.tanh = nullptr; }
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
    g_host_tables = false;
    const bool stream = !getenv("K3X_NOSTREAM");
    struct R { long gid; int tg, td; size_t off; };
    std::vector<R> idx; { FILE * f = fopen(idxp, "r"); if (!f) { perror(idxp); return 1; } R r; while (fscanf(f, "%ld %d %d %zu", &r.gid, &r.tg, &r.td, &r.off) == 4) idx.push_back(r); fclose(f); }
    std::vector<Engine *> eng; auto get = [&](int tg, int td) -> Engine * { for (auto * e : eng) if (e->tg == tg && e->td == td) return e; eng.push_back(new Engine(tg, td, 0)); eng.back()->spin = true; load_mufu(*eng.back()); return eng.back(); };
    std::unordered_map<long, std::pair<Engine *, const uint8_t *>> map;
    FILE * f = fopen(binp, "rb"); if (!f) { perror(binp); return 1; }
    const size_t CH = (size_t) 1 << 30; uint8_t * chunk = nullptr; size_t used = CH; std::vector<uint8_t> buf, rbuf;
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
        rbuf.resize(e->prec); e->repack(buf.data(), rbuf.data());
        e->q.memcpy(chunk + used, rbuf.data(), e->prec).wait();
        map[r.gid] = {e, chunk + used}; used += e->prec; tot += e->prec;
    }
    fclose(f);
    for (auto & v : tab_host) std::vector<char>().swap(v);   // host copies only serve the host reference
    fprintf(stderr, "loaded %zu of %zu records (%.1f GB on the GPU) in %.1f s, %zu engine kinds\n", map.size(), idx.size(), tot / 1e9, (now_ms() - t0) / 1e3, eng.size());
    if (const char * so = getenv("K3X_SETOUT")) {   // "layer expert" of the served records, for LLAMA_EXPERT_STORE_REMOTE_SET
        FILE * f = fopen(so, "w");
        for (auto & r : idx) if (map.count(r.gid)) fprintf(f, "%ld %ld\n", r.gid / 896, r.gid % 896);
        fclose(f);
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
    std::vector<float> nanrows(16 * D, NAN), xin(D); long zero_rows = 0;
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
            bool okw = true;
            // A dropped GPU submission (i915 EXECBUFFER2 EFAULT under a very large residency set) leaves an all-zero
            // row; a real expert row is never all zero, so such a row is sent as NaN and the client reads it locally.
            auto guard = [&](int j) { const float * r = e->rows + (size_t) j * D; for (int i = 0; i < D; i++) if (r[i] != 0.0f) return r;
                                      zero_rows++; return (const float *) nanrows.data(); };
            if (stream) { std::function<void(int)> snk = [&](int j) { if (okw) okw = wr_all(c, guard(j), D * 4); }; e->run(k, &snk); }
            else { e->run(k); for (int j = 0; j < k && okw; j++) okw = wr_all(c, guard(j), D * 4); }
            if (!okw) break;
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
            if (en.prof) { printf("  kernels (ms avg): quant %.3f gate/up %.3f situ+quant %.3f down %.3f\n", en.kt[0] / en.nrun, en.kt[1] / en.nrun, en.kt[2] / en.nrun, en.kt[3] / en.nrun); for (double & v : en.kt) v = 0; en.nrun = 0; }
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
