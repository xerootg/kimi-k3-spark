// k3x_mufu.h -- NVIDIA MUFU.RCP / MUFU.EX2 / MUFU.TANH raw results from agent A's exhaustive tables (ref/mufu/),
// held packed (k3_ptab, ~1.1 byte per entry instead of 4, exact; see k3x_math.h).
#pragma once
// MUFU.RCP: rcp.bin = 2^23 x u32, entry m = bits of MUFU.RCP(1.m) (depends only on the mantissa); exponent applied exactly.
// Denormal input reads as signed zero (-> signed inf); a denormal result is flushed to signed zero.
static inline float nv_rcp(float x, const k3_mufu_tabs & M) {
    const uint32_t u = k3_f2u(x), sg = u & 0x80000000u, ex = (u >> 23) & 0xff, m = u & 0x7fffffu;
    if (ex == 0) return k3_u2f(sg | 0x7f800000u);                 // 0 or denormal -> inf
    if (ex == 255) return m ? k3_u2f(0x7fffffffu) : k3_u2f(sg);   // NaN -> canonical NaN, inf -> 0
    if (!M.rcp.n) return 1.0f / x;
    const uint32_t r = k3_ptab_get(M.rcp, m);
    const int e = (int) ((r >> 23) & 0xff) + 127 - (int) ex;
    if (e <= 0) return k3_u2f(sg);
    if (e >= 255) return k3_u2f(sg | 0x7f800000u);
    return k3_u2f(sg | ((uint32_t) e << 23) | (r & 0x7fffffu));
}
// MUFU.TANH: tanh.bin entry i = MUFU.TANH(bits 0x39800000 + i), |x| in [2^-12, 8); odd-symmetric (A).
// Below 2^-12 the result is x; at or above the table end the result is +-1 (ASSUMPTION, checked by validate).
static inline float nv_tanh(float x, const k3_mufu_tabs & M) {
    x = k3_ftz(x);
    const uint32_t u = k3_f2u(x), sg = u & 0x80000000u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return k3_u2f(0x7fffffffu);
    if (a < 0x39800000u) return x;
    if (!M.tanh.n) return tanhf(x);
    const uint32_t i = a - 0x39800000u;
    if (i >= M.n_tanh) return k3_u2f(sg | 0x3f800000u);
    return k3_u2f(sg | k3_ptab_get(M.tanh, i));
}
// MUFU.EX2 (same layout and edge rules as tools/bitexact/mufu/mufu_emu.h; confirmed by validate):
//   ex2p.bin entry i = MUFU.EX2(+x), bits(x) = 0x33800000 + i, x in [2^-24, 128) (31 binades); below -> 1.0, >= 128 -> inf
//   ex2n.bin entry i = MUFU.EX2(-x), bits(x) = 0x33000000 + i, x from 2^-25 up; below -> 1.0, past the end -> +0 (FTZ)
static inline float nv_ex2(float x, const k3_mufu_tabs & M) {
    x = k3_ftz(x);
    const uint32_t u = k3_f2u(x), a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return k3_u2f(0x7fffffffu);
    if (!(u & 0x80000000u)) {
        if (a < 0x33800000u) return 1.0f;
        if (!M.ex2p.n) return exp2f(x);
        const uint32_t i = a - 0x33800000u;
        return i < M.n_ex2p ? k3_u2f(k3_ptab_get(M.ex2p, i)) : k3_u2f(0x7f800000u);
    }
    if (a < 0x33000000u) return 1.0f;
    if (!M.ex2n.n) return exp2f(x);
    const uint32_t i = a - 0x33000000u;
    return i < M.n_ex2n ? k3_u2f(k3_ptab_get(M.ex2n, i)) : 0.0f;
}
