// k3x_math.h -- bit-exact emulation of the Spark's (GB10, sm_121) ggml-cuda MMVQ arithmetic for one decode token.
// Shared by the SYCL device code and the host reference (same source, compiled with -ffp-contract=off and a precise FP model).
// Every float op is written out in CUDA's order; FMAs that nvcc forms are written as explicit fmaf() calls.
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/sycl.hpp>
#define K3_FMA(a, b, c) sycl::fma((a), (b), (c))
#else
#define K3_FMA(a, b, c) std::fma((a), (b), (c))
#endif


static constexpr int K3_D = 3584, K3_F = 3072, K3_QK = 256;
enum { T_IQ2_XXS = 16, T_IQ1_S = 19 };

static inline uint32_t k3_f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float k3_u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// IEEE half -> float (exact) and float -> half round-to-nearest-even (__float2half_rn), bit level so host == device
static inline float k3_h2f(uint16_t h) {
    uint32_t s = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        if (m == 0) return k3_u2f(s);
        float f = (float) m * 5.9604644775390625e-8f; // m * 2^-24, exact
        return k3_u2f(k3_f2u(f) | s);
    }
    if (e == 31) return k3_u2f(s | 0x7f800000 | (m << 13));
    return k3_u2f(s | ((e + 112) << 23) | (m << 13));
}
static inline uint16_t k3_f2h(float f) {
    uint32_t x = k3_f2u(f), s = (x >> 16) & 0x8000, ax = x & 0x7fffffff;
    if (ax >= 0x7f800000) return s | 0x7c00 | (ax > 0x7f800000 ? 0x200 : 0);
    if (ax >= 0x477ff000) return s | 0x7c00;           // rounds to >= 65520 -> inf
    if (ax < 0x38800000) {                             // subnormal half (or zero)
        if (ax < 0x33000000) return s;                 // < 2^-25 -> 0 (2^-25 exactly ties to even 0)
        uint32_t e = ax >> 23, m = (ax & 0x7fffff) | 0x800000;
        uint32_t sh = 126 - e;                         // shift so that result is in units of 2^-24
        uint32_t q = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        return s | q;
    }
    uint32_t e = (ax >> 23) - 112, m = ax & 0x7fffff;
    uint32_t q = (e << 10) | (m >> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (q & 1))) q++;
    return s | q;
}

static inline int k3_dp4a(int a, int b, int c) {
    for (int i = 0; i < 4; i++) c += (int) (int8_t) (a >> (8 * i)) * (int) (int8_t) (b >> (8 * i));
    return c;
}
static inline int k3_get_int_b2(const uint8_t * x, int i32) {
    uint16_t lo, hi; memcpy(&lo, x + 4 * i32, 2); memcpy(&hi, x + 4 * i32 + 2, 2);
    return (int) ((uint32_t) lo | ((uint32_t) hi << 16));
}
static inline int k3_get_int_b4(const int8_t * x, int i32) { int v; memcpy(&v, x + 4 * i32, 4); return v; }

// q8_1 block as CUDA lays it out: half d, half s (=sum), int8 qs[32]
struct alignas(4) k3_q8_1 { uint16_t d, s; int8_t qs[32]; };

// ---- FTZ arithmetic as on the Spark (every FADD/FMUL/FFMA is .FTZ: denormal in -> signed 0, denormal out -> signed 0)
static inline float k3_ftz(float x) { return (k3_f2u(x) & 0x7f800000u) == 0 ? k3_u2f(k3_f2u(x) & 0x80000000u) : x; }
static inline float fadd_z(float a, float b) { return k3_ftz(k3_ftz(a) + k3_ftz(b)); }
static inline float fmul_z(float a, float b) { return k3_ftz(k3_ftz(a) * k3_ftz(b)); }
static inline float ffma_z(float a, float b, float c) { return k3_ftz(K3_FMA(k3_ftz(a), k3_ftz(b), k3_ftz(c))); }

// ---- NVIDIA MUFU (raw special-function unit results); implemented in k3x_mufu.h from agent A's tables
struct k3_mufu_tabs { const uint32_t * rcp; const uint32_t * ex2p; const uint32_t * ex2n; const uint32_t * tanh; uint32_t n_ex2p, n_ex2n, n_tanh; };
#include "k3x_mufu.h"

// q8_1 quantization of 32 floats (SPEC section 2)
static inline void k3_quantize_block(const float * xin, k3_q8_1 * y, const k3_mufu_tabs & M) {
    float x[32], a[32], s[32];
    for (int i = 0; i < 32; i++) { x[i] = k3_ftz(xin[i]); a[i] = fabsf(x[i]); s[i] = x[i]; }
    for (int off = 16; off > 0; off >>= 1)
        for (int i = 0; i < off; i++) { a[i] = fmaxf(a[i], a[i + off]); s[i] = fadd_z(s[i], s[i + off]); }
    const float amax = a[0], sum = s[0];
    const float d = fmul_z(amax, k3_u2f(0x3C010204u));
    if (amax != 0.0f) {
        const float r = nv_rcp(d, M);
        // q = trunc(RZ(v + copysign(0.5, v))) == round-half-away(v) exactly (integers are representable, RZ never crosses one)
        for (int i = 0; i < 32; i++) y->qs[i] = (int8_t) (int) roundf(fmul_z(r, x[i]));
    } else {
        for (int i = 0; i < 32; i++) y->qs[i] = 0;
    }
    y->d = k3_f2h(d);
    y->s = k3_f2h(sum);
}

// float tail of vec_dot_iq1_s_q8_1, already fused with the +0 accumulator: p_t (SPEC section 3)
static inline float k3_iq1s_finish(float dhf, int qh, int sumi, float dsx, float dsy) {
    const float dW = fmul_z(dhf, (float) (((qh >> 11) & 0x0E) + 1));
    const float delta = (qh & 0x8000) ? -1.125f : -0.875f;   // fma(-(qh&0x8000), 2^-17, -0.875): exact
    const float inner = ffma_z(dsx, (float) sumi, fmul_z(dsy, delta));
    return ffma_z(dW, inner, 0.0f);
}
// float tail of vec_dot_iq2_xxs_q8_1 (sumi already * ls / 8), fused with the +0 accumulator
static inline float k3_iq2xxs_finish(float dhf, int sumi, float q8df) {
    const float dW = fmul_z(dhf, q8df);
    return ffma_z(dW, (float) sumi, 0.0f);
}

// iq1s_grid_gpu / iq2xxs_grid are passed in (USM on device)
// IQ1_S: 50-byte block {half d; uint8 qs[32]; uint16 qh[8]}; lane iqs in 0..7 uses q8 block iqs
static inline float k3_vdot_iq1_s(const uint8_t * blk, const k3_q8_1 * q8, int iqs, const uint32_t * grid_tab) {
    uint16_t dh; memcpy(&dh, blk, 2);
    const uint8_t * qsb = blk + 2;
    const uint8_t * qhb = blk + 34;
    const int qs_packed = k3_get_int_b2(qsb, iqs);
    uint16_t qh16; memcpy(&qh16, qhb + 2 * iqs, 2);
    const int qh = qh16;
    const k3_q8_1 * b8 = q8 + iqs;
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qsv = (qs_packed >> (8 * (l0 / 2))) & 0xff;
        const int grid = (int) grid_tab[qsv | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = k3_get_int_b4(b8->qs, l0 + 0);
        const int u1 = k3_get_int_b4(b8->qs, l0 + 1);
        sumi = k3_dp4a(grid0, u0, sumi);
        sumi = k3_dp4a(grid1, u1, sumi);
    }
    return k3_iq1s_finish(k3_h2f(dh), qh, sumi, k3_h2f(b8->d), k3_h2f(b8->s));
}

// IQ2_XXS: 66-byte block {half d; uint16 qs[32]}; lane l -> iqs = 2l, q8 block iqs/2 = l
static inline float k3_vdot_iq2_xxs(const uint8_t * blk, const k3_q8_1 * q8, int iqs, const uint64_t * grid_tab) {
    uint16_t dh; memcpy(&dh, blk, 2);
    const uint8_t * qs = blk + 2;
    const int q2 = k3_get_int_b2(qs, iqs);
    const uint32_t aux32 = (uint32_t) k3_get_int_b2(qs, iqs + 1);
    const k3_q8_1 * b8 = q8 + iqs / 2;
    int sumi = 0;
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint64_t gp = grid_tab[(q2 >> (8 * (k0 / 2))) & 0xff];
        const uint32_t gx = (uint32_t) gp, gy = (uint32_t) (gp >> 32);
        const uint8_t v = (uint8_t) (aux32 >> (7 * k0 / 2));
        uint32_t pc = 0; for (int i = 0; i < 8; i++) pc += (v >> i) & 1;
        const uint32_t sgn = ((uint32_t) v ^ ((pc & 1) << 7)) * 0x01010101u;
        uint32_t s0 = 0, s1 = 0;
        for (int i = 0; i < 4; i++) {
            if ((sgn & 0x08040201u) & (0xffu << (8 * i))) s0 |= 0xffu << (8 * i);
            if ((sgn & 0x80402010u) & (0xffu << (8 * i))) s1 |= 0xffu << (8 * i);
        }
        uint32_t g0 = 0, g1 = 0;
        for (int i = 0; i < 4; i++) {
            const uint32_t a0 = ((gx ^ s0) >> (8 * i)) & 0xff, b0 = (s0 >> (8 * i)) & 0xff;
            const uint32_t a1 = ((gy ^ s1) >> (8 * i)) & 0xff, b1 = (s1 >> (8 * i)) & 0xff;
            g0 |= ((a0 - b0) & 0xff) << (8 * i);
            g1 |= ((a1 - b1) & 0xff) << (8 * i);
        }
        sumi = k3_dp4a((int) g0, k3_get_int_b4(b8->qs, k0 + 0), sumi);
        sumi = k3_dp4a((int) g1, k3_get_int_b4(b8->qs, k0 + 1), sumi);
    }
    const int ls = (int) (aux32 >> 27 | 1);
    sumi = sumi * ls / 8;
    return k3_iq2xxs_finish(k3_h2f(dh), sumi, k3_h2f(b8->d));
}

static inline int k3_blk_bytes(int t) { return t == T_IQ1_S ? 50 : 66; }

// lane result of virtual CUDA thread (kbx, l) for a row of type t: blk = row + kbx*bytes, q8 = y + kbx*8
static inline float k3_vdot(int t, const uint8_t * blk, const k3_q8_1 * q8b, int l, const uint32_t * g1, const uint64_t * g2) {
    return t == T_IQ1_S ? k3_vdot_iq1_s(blk, q8b, l, g1) : k3_vdot_iq2_xxs(blk, q8b, 2 * l, g2);
}

// mul_mat_vec_q<.., ncols_dst=1, small_k> per-row reduction: 4 warps x 32 lanes, v[tid] = lane partial (idle lanes absent)
// p[128] holds p_t = fma(dW, inner, +0) for active tids and +0.0f for idle ones.
static inline float k3_row_reduce(const float * p) {
    float s[32];
    for (int x = 0; x < 32; x++) {
        float t = p[x];               // FADD.FTZ in CUDA; no denormal can arise here (see k3x.cpp iq1s_tail note)
        t = t + p[32 + x];
        t = t + p[64 + x];
        t = t + p[96 + x];
        s[x] = t;
    }
    for (int off = 16; off > 0; off >>= 1)
        for (int x = 0; x < off; x++) s[x] = s[x] + s[x + off];
    return s[0];
}
