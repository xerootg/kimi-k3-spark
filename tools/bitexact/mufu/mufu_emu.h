// Table-based emulation of the GB10's (sm_121a) MUFU.TANH, MUFU.EX2, MUFU.RCP, bit-exact over all 2^32 inputs
// (verified exhaustively by mufu_tables.cu verify). Plain C99 / C++ / CUDA (define MUFU_EMU_FN to add qualifiers).
//
// Tables (little-endian uint32 = output float bits), produced by `mufu_tables gen` on the GB10:
//   rcp.bin     2^23 entries  MUFU.RCP(1.m), m = 23-bit mantissa                         32 MiB
//   tanh.bin    15 * 2^23     MUFU.TANH(+x) for biased exponent 115..129 (|x| in [2^-12, 8))   480 MiB
//   ex2p.bin    31 * 2^23     MUFU.EX2(+x)  for biased exponent 103..133 (x in [2^-24, 128))   992 MiB
//   ex2n.bin    33 * 2^23     MUFU.EX2(-x)  for biased exponent 102..134 (x in (-256, -2^-25])  1056 MiB
// Everything outside those ranges follows the closed-form rules below.
#ifndef MUFU_EMU_H
#define MUFU_EMU_H
#include <stdint.h>
#ifndef MUFU_EMU_FN
#define MUFU_EMU_FN static inline
#endif

#define MUFU_TANH_E0 115
#define MUFU_TANH_E1 129
#define MUFU_EX2P_E0 103
#define MUFU_EX2P_E1 133
#define MUFU_EX2N_E0 102
#define MUFU_EX2N_E1 134
#define MUFU_NAN 0x7fffffffu

typedef struct { const uint32_t * rcp, * tanh_, * ex2p, * ex2n; } mufu_tables;

MUFU_EMU_FN uint32_t mufu_rcp_bits(const mufu_tables * T, uint32_t u) {
    const uint32_t s = u & 0x80000000u, E = (u >> 23) & 255u, m = u & 0x7fffffu;
    if (E == 255u) return m ? MUFU_NAN : s;            // NaN -> canonical NaN, +-inf -> +-0
    if (E == 0u) return s | 0x7f800000u;               // +-0 and denormals (flushed) -> +-inf
    const uint32_t t = T->rcp[m];
    const int e = (int) ((t >> 23) & 255u) + 127 - (int) E;
    if (e < 1) return s;                               // result would be denormal: flushed to +-0
    return s | ((uint32_t) e << 23) | (t & 0x7fffffu);
}

MUFU_EMU_FN uint32_t mufu_tanh_bits(const mufu_tables * T, uint32_t u) {
    const uint32_t s = u & 0x80000000u, a = u & 0x7fffffffu, E = a >> 23;
    if (a > 0x7f800000u) return MUFU_NAN;
    if (E < MUFU_TANH_E0) return u;                    // |x| < 2^-12, incl. +-0 and denormals (kept): tanh(x) = x
    if (E <= MUFU_TANH_E1) return s | T->tanh_[((E - MUFU_TANH_E0) << 23) | (a & 0x7fffffu)];
    return s | 0x3f800000u;                            // |x| >= 8 (and inf): +-1
}

MUFU_EMU_FN uint32_t mufu_ex2_bits(const mufu_tables * T, uint32_t u) {
    const uint32_t a = u & 0x7fffffffu, E = a >> 23;
    if (a > 0x7f800000u) return MUFU_NAN;
    if (E == 0u) return 0x3f800000u;                   // +-0, denormals -> 1
    if (!(u >> 31)) {
        if (E < MUFU_EX2P_E0) return 0x3f800000u;
        if (E <= MUFU_EX2P_E1) return T->ex2p[((E - MUFU_EX2P_E0) << 23) | (a & 0x7fffffu)];
        return 0x7f800000u;                            // x >= 128: +inf
    }
    if (E < MUFU_EX2N_E0) return 0x3f800000u;
    if (E <= MUFU_EX2N_E1) return T->ex2n[((E - MUFU_EX2N_E0) << 23) | (a & 0x7fffffu)];
    return 0u;                                         // x <= -256: +0
}
#endif
