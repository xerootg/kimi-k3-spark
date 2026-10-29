// k3cpu.cpp -- see k3cpu.h and README.md. Build flags (build_k3cpu.sh): -O2 -std=c++17 -ffp-contract=off
// -fno-fast-math -march=x86-64-v3 -mavxvnni. Every float op of SPEC.md is written out below in CUDA's order; the
// workers run with MXCSR DAZ|FTZ, which is the .FTZ of every FADD/FMUL/FFMA (denormal in -> signed 0, out -> signed 0;
// x86 detects tininess after rounding). Integer dot products are reordered freely (exact).
#include "k3cpu.h"
#include "../sycl/k3x_act.h"
#include <immintrin.h>
#include <atomic>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <climits>
#include <algorithm>
#include <cmath>
#include <string>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <unistd.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_SYCL   // plain static tables: iq1s_grid_gpu (CUDA layout) and iq2xxs_grid
#include "../sycl/ggml-common.h"

static constexpr int D = 3584, F = 3072;
static constexpr int NB_X = D / 256, NB_H = F / 256;          // 14, 12 q-blocks of 256 per row
static constexpr int CH_GU = 32, CH_DN = 32;                 // rows per task (gate/up chunk = one q8_1 block of h)
static constexpr int NGU = F / CH_GU, NDN = D / CH_DN;       // 96, 112 tasks per expert
static constexpr int NQX = 8, QX_BLK = D / 32 / NQX;         // q8_1(x): 8 tasks of 14 blocks, before any gate/up task

static inline uint64_t now_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return (uint64_t) t.tv_sec * 1000000000ull + t.tv_nsec; }
static inline void set_daz_ftz() { _mm_setcsr(_mm_getcsr() | 0x8040); }

size_t k3cpu_rec_bytes(int tg, int tu, int td, int layout) {
    auto rb = [layout](int t, int k) { return (size_t) (k / 256) * (layout == K3CPU_V7 ? (t == K3CPU_T_IQ1_S ? 52 : 68) : (t == K3CPU_T_IQ1_S ? 50 : 66)); };
    return rb(tg, D) * F + rb(tu, D) * F + rb(td, F) * D;
}

// ------------------------------------------------------------------------------------------------ q8_1 activations
// Layout for the VNNI dots: per q-block kb (256 values = q8_1 blocks 8kb..8kb+7), qt[kb][r][j] = dword r (values
// 4r..4r+3) of q8_1 block 8kb+j, so one 32-byte register r holds the same 4 positions of all 8 sub-blocks and a
// vpdpbusd lane j accumulates sub-block j's dot: no horizontal sums. dsx = h2f(ds.x), dym/dyp = h2f(ds.y) * -0.875 /
// -1.125 (the two IQ1_S delta products, exact).
// q64 is the same bytes for the scalar-lookup kernel: per kb, group i, half h: q8_1 blocks 8kb+4h..8kb+4h+3, values
// 8i..8i+7 of each (32 bytes), matching one register of four 64-bit grid entries (lanes = j, low/high nibbles).
// For v7 records (4 rows per pass, lanes = (sub-block j = 2sp + h, row)): qv[kb][sp][r] = dword r of q8_1 block
// 8kb+2sp (lanes 0-3) and 8kb+2sp+1 (lanes 4-7), each repeated for the 4 rows; ds*4 likewise per lane. (The scalar-load
// kernel broadcasts its 8 bytes per row straight from q64.)
struct alignas(64) QX {
    alignas(64) int8_t qt[NB_X * 256];
    alignas(64) int8_t q64[NB_X * 256];
    alignas(64) float dsx[NB_X * 8];
    alignas(64) float dym[NB_X * 8];
    alignas(64) float dyp[NB_X * 8];
    alignas(64) int8_t qv[NB_X * 4 * 8 * 32];
    alignas(64) float dsx4[NB_X * 32];
    alignas(64) float dym4[NB_X * 32];
    alignas(64) float dyp4[NB_X * 32];
    bool v7 = false;   // also fill the v7 layouts
};
// all layouts of q8_1 block b from its 32 values and float scales
static inline void q8_put(QX & q, int b, const int8_t * qs, float dx, float ym, float yp) {
    const int kb = b >> 3, j = b & 7;
    int8_t * qt = q.qt + kb * 256 + 4 * j;
    for (int r = 0; r < 8; r++) memcpy(qt + 32 * r, qs + 4 * r, 4);
    int8_t * o = q.q64 + kb * 256 + ((j >> 2) & 1) * 32 + (j & 3) * 8;
    for (int i = 0; i < 4; i++) memcpy(o + 64 * i, qs + 8 * i, 8);
    q.dsx[b] = dx; q.dym[b] = ym; q.dyp[b] = yp;
    if (!q.v7) return;
    const int sp = j >> 1, h = j & 1;
    int8_t * v = q.qv + (size_t) (kb * 4 + sp) * 8 * 32 + h * 16;
    for (int r = 0; r < 8; r++) for (int row = 0; row < 4; row++) memcpy(v + r * 32 + row * 4, qs + 4 * r, 4);
    for (int row = 0; row < 4; row++) {
        const int l = (kb * 4 + sp) * 8 + h * 4 + row;
        q.dsx4[l] = dx; q.dym4[l] = ym; q.dyp4[l] = yp;
    }
}
static inline void q8_store(QX & q, int b, const int8_t * qs, float d, float sum) {
    const float dx = k3_h2f(k3_f2h(d)), dy = k3_h2f(k3_f2h(sum));
    q8_put(q, b, qs, dx, fmul_z(dy, -0.875f), fmul_z(dy, -1.125f));
}

// SPEC 2 literally (k3ref's oracle: NaN-ignoring FMNMX, FADD.RZ, saturating F2I, low byte), for the blocks the vector
// path does not cover: a NaN or inf anywhere, or d = amax/127 flushed to zero.
static float fmax_nv(float a, float b) { a = k3_ftz(a); b = k3_ftz(b); if (a != a) return b; if (b != b) return a; return a > b ? a : b; }
static float fadd_rz(float a, float b) {
    a = k3_ftz(a); b = k3_ftz(b);
    const double s = (double) a + (double) b;
    float r = (float) s;
    if (std::isfinite(r) && std::fabs((double) r) > std::fabs(s)) r = std::nextafter(r, 0.0f);
    return k3_ftz(r);
}
static void q8_block_ref(const float * xb, QX & q, int b, const k3_mufu_tabs & M) {
    float am[32], sm[32];
    for (int i = 0; i < 32; i++) { am[i] = fadd_z(std::fabs(xb[i]), -0.0f); sm[i] = xb[i]; }
    for (int off = 16; off > 0; off >>= 1) {
        float na[32], ns[32];
        for (int i = 0; i < 32; i++) { na[i] = fmax_nv(am[i], am[i ^ off]); ns[i] = fadd_z(sm[i], sm[i ^ off]); }
        memcpy(am, na, sizeof am); memcpy(sm, ns, sizeof sm);
    }
    const float amax = am[0], sum = sm[0], d = fmul_z(amax, k3_u2f(0x3C010204u));
    int8_t qs[32];
    for (int i = 0; i < 32; i++) {
        int32_t v = 0;
        if (!(amax == 0.0f)) {   // FSETP.NEU: unordered -> true
            const float r = nv_rcp(d, M), w = fmul_z(r, xb[i]);
            const float t = std::trunc(fadd_rz(w, k3_u2f(0x3f000000u | (k3_f2u(w) & 0x80000000u))));
            v = t != t ? 0 : t >= 2147483648.0f ? INT32_MAX : t <= -2147483648.0f ? INT32_MIN : (int32_t) t;
        }
        qs[i] = (int8_t) (uint8_t) (v & 0xff);
    }
    q8_store(q, b, qs, d, sum);
}

// SPEC 2 for one block of 32 (MXCSR DAZ|FTZ must be on): writes block b = 8kb + j of q
static inline void q8_block(const float * xb, QX & q, int b, const k3_mufu_tabs & M) {
    const __m256 X0 = _mm256_loadu_ps(xb), X1 = _mm256_loadu_ps(xb + 8), X2 = _mm256_loadu_ps(xb + 16), X3 = _mm256_loadu_ps(xb + 24);
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)), nz = _mm256_set1_ps(-0.0f);
    // |x| + (-0): FADD.FTZ flushes a denormal |x| to +0 (DAZ); max is order-independent
    __m256 A = _mm256_max_ps(_mm256_max_ps(_mm256_add_ps(_mm256_and_ps(X0, absm), nz), _mm256_add_ps(_mm256_and_ps(X1, absm), nz)),
                             _mm256_max_ps(_mm256_add_ps(_mm256_and_ps(X2, absm), nz), _mm256_add_ps(_mm256_and_ps(X3, absm), nz)));
    __m128 a4 = _mm_max_ps(_mm256_castps256_ps128(A), _mm256_extractf128_ps(A, 1));
    a4 = _mm_max_ps(a4, _mm_movehl_ps(a4, a4));
    a4 = _mm_max_ss(a4, _mm_shuffle_ps(a4, a4, 1));
    const float amax = _mm_cvtss_f32(a4);
    // butterfly sum = tree x[j]+x[j+16], +[j+8], +[j+4], +[j+2], +[j+1]
    const __m256 T = _mm256_add_ps(_mm256_add_ps(X0, X2), _mm256_add_ps(X1, X3));
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(T), _mm256_extractf128_ps(T, 1));
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
    const float sum = _mm_cvtss_f32(s4);
    const float d = amax * k3_u2f(0x3C010204u);
    // vector path: everything finite and d normal (or all zero); anything else takes the literal path
    if (!(std::fabs(sum) <= 3.4e38f) || !(amax <= 3.4e38f) || (amax != 0.0f && d == 0.0f)) { q8_block_ref(xb, q, b, M); return; }
    alignas(32) int8_t qs[32];
    if (amax != 0.0f) {
        const __m256 R = _mm256_set1_ps(nv_rcp(d, M));
        const __m256 half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f), sgn = _mm256_set1_ps(-0.0f);
        __m256i Q[4];
        const __m256 Xs[4] = {X0, X1, X2, X3};
        for (int i = 0; i < 4; i++) {
            // v = fmul(r, x); q = trunc(RZ(v + copysign(0.5, v))) = round half away from zero (exact, |v| <= 127)
            const __m256 v = _mm256_mul_ps(R, Xs[i]);
            const __m256 t = _mm256_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            const __m256 fr = _mm256_andnot_ps(sgn, _mm256_sub_ps(v, t));
            const __m256 inc = _mm256_and_ps(_mm256_cmp_ps(fr, half, _CMP_GE_OQ), _mm256_or_ps(one, _mm256_and_ps(v, sgn)));
            Q[i] = _mm256_cvttps_epi32(_mm256_add_ps(t, inc));
        }
        const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(Q[0], Q[1]), _mm256_packs_epi32(Q[2], Q[3]));
        _mm256_store_si256((__m256i *) qs, _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
    } else {
        memset(qs, 0, sizeof qs);
    }
    q8_store(q, b, qs, d, sum);
}

// ------------------------------------------------------------------------------------------------ row kernels
static const __m256i * Qreg(const QX & q, int kb) { return (const __m256i *) (q.qt + kb * 256); }

// IQ1_S block (50 B: half d, qs[32], qh[8]) against q-block kb: p_t for the 8 virtual threads j of block kb (SPEC 3)
__attribute__((always_inline)) static inline __m256 blk_iq1_s(const uint8_t * b, const QX & q, int kb) {
    const __m256i QS = _mm256_loadu_si256((const __m256i *) (b + 2));                      // dword j = qs[4j..4j+3]
    const __m256i QH = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) (b + 34))); // lane j = qh[j]
    const __m256i m7 = _mm256_set1_epi32(0x700), mlo = _mm256_set1_epi32(0x0F0F0F0F), mhi = _mm256_set1_epi32((int) 0xF0F0F0F0u);
    const int * G = (const int *) iq1s_grid_gpu;
    // grid index of group i of sub-block j: qs[4j+i] (byte i of dword j, zero-extended by one vpshufb) | ((qh[j] >> 3i) & 7) << 8
#define K3_SEL(i) _mm256_setr_epi8(i, -1, -1, -1, 4 + i, -1, -1, -1, 8 + i, -1, -1, -1, 12 + i, -1, -1, -1, i, -1, -1, -1, 4 + i, -1, -1, -1, 8 + i, -1, -1, -1, 12 + i, -1, -1, -1)
    const __m256i i0 = _mm256_or_si256(_mm256_shuffle_epi8(QS, K3_SEL(0)), _mm256_and_si256(_mm256_slli_epi32(QH, 8), m7));
    const __m256i i1 = _mm256_or_si256(_mm256_shuffle_epi8(QS, K3_SEL(1)), _mm256_and_si256(_mm256_slli_epi32(QH, 5), m7));
    const __m256i i2 = _mm256_or_si256(_mm256_shuffle_epi8(QS, K3_SEL(2)), _mm256_and_si256(_mm256_slli_epi32(QH, 2), m7));
    const __m256i i3 = _mm256_or_si256(_mm256_shuffle_epi8(QS, K3_SEL(3)), _mm256_and_si256(_mm256_srli_epi32(QH, 1), m7));
#undef K3_SEL
#ifdef K3_TIMING_NOGATHER   // timing experiment only: wrong results
    (void) G; const __m256i g0 = i0, g1 = i1, g2 = i2, g3 = i3;
#else
    const __m256i g0 = _mm256_i32gather_epi32(G, i0, 4), g1 = _mm256_i32gather_epi32(G, i1, 4);
    const __m256i g2 = _mm256_i32gather_epi32(G, i2, 4), g3 = _mm256_i32gather_epi32(G, i3, 4);
#endif
    const __m256i * Q = Qreg(q, kb);
    // grid nibbles are 0..2: low ones (u8) against values 8i..8i+3, high ones left in place (u8 0, 16, 32) against
    // 8i+4..8i+7, so a1 = 16 * their dot; vpdpbusd (u8 x s8, 32-bit sums, no saturation) is exact either way
    __m256i a0 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_and_si256(g0, mlo), Q[0]);
    __m256i a1 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_and_si256(g0, mhi), Q[1]);
    a0 = _mm256_dpbusd_avx_epi32(a0, _mm256_and_si256(g1, mlo), Q[2]);
    a1 = _mm256_dpbusd_avx_epi32(a1, _mm256_and_si256(g1, mhi), Q[3]);
    a0 = _mm256_dpbusd_avx_epi32(a0, _mm256_and_si256(g2, mlo), Q[4]);
    a1 = _mm256_dpbusd_avx_epi32(a1, _mm256_and_si256(g2, mhi), Q[5]);
    a0 = _mm256_dpbusd_avx_epi32(a0, _mm256_and_si256(g3, mlo), Q[6]);
    a1 = _mm256_dpbusd_avx_epi32(a1, _mm256_and_si256(g3, mhi), Q[7]);
    const __m256 sumi = _mm256_cvtepi32_ps(_mm256_add_epi32(a0, _mm256_srai_epi32(a1, 4)));
    const __m256 sc = _mm256_cvtepi32_ps(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(QH, 11), _mm256_set1_epi32(0x0E)), _mm256_set1_epi32(1)));
    uint16_t dh; memcpy(&dh, b, 2);
    const __m256 dW = _mm256_mul_ps(_mm256_set1_ps(_cvtsh_ss(dh)), sc);                           // FMUL.FTZ
    const __m256 yd = _mm256_blendv_ps(_mm256_load_ps(q.dym + 8 * kb), _mm256_load_ps(q.dyp + 8 * kb), _mm256_castsi256_ps(_mm256_slli_epi32(QH, 16)));
    const __m256 inner = _mm256_fmadd_ps(_mm256_load_ps(q.dsx + 8 * kb), sumi, yd);              // FFMA.FTZ
    return _mm256_fmadd_ps(dW, inner, _mm256_setzero_ps());                                       // FFMA.FTZ dW*inner + 0
}

// Same block with the grid looked up by scalar loads instead of vpgatherdd (faster on the E-cores, where a ymm gather
// costs ~45 cycles): indices in GPRs, 64-bit entries of g64 = {grid & 0x0F0F0F0F, (grid >> 4) & 0x0F0F0F0F} (the 8
// values of a group as bytes), four per register (lanes j, low/high nibble dwords) against q64; then lane pairs are added.
alignas(64) static uint64_t g64[2048];
static void init_g64() { for (int i = 0; i < 2048; i++) { const uint32_t g = iq1s_grid_gpu[i]; g64[i] = (uint64_t) (g & 0x0F0F0F0Fu) | (uint64_t) ((g >> 4) & 0x0F0F0F0Fu) << 32; } }
__attribute__((always_inline)) static inline __m256 blk_iq1_s_sc(const uint8_t * b, const QX & q, int kb) {
    const uint8_t * qs = b + 2, * qhp = b + 34;
    const __m256i * Q = (const __m256i *) (q.q64 + kb * 256);
    __m256i acc[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
#pragma GCC unroll 2
    for (int h = 0; h < 2; h++) {
        uint64_t W[4];   // per j': the four 11-bit grid indices as 16-bit fields, i = 0..3
#pragma GCC unroll 4
        for (int jj = 0; jj < 4; jj++) {
            const int j = 4 * h + jj;
            uint32_t q4; memcpy(&q4, qs + 4 * j, 4);
            uint16_t hw; memcpy(&hw, qhp + 2 * j, 2);
            // qs[4j+i] -> bits 16i..16i+7, ((qh >> 3i) & 7) -> bits 16i+8..16i+10
            W[jj] = _pdep_u64(q4, 0x00FF00FF00FF00FFull) | _pdep_u64(hw, 0x0700070007000700ull);
        }
#pragma GCC unroll 4
        for (int i = 0; i < 4; i++) {
            const __m128i lo = _mm_set_epi64x((long long) g64[(W[1] >> 16 * i) & 0xFFFF], (long long) g64[(W[0] >> 16 * i) & 0xFFFF]);
            const __m128i hi = _mm_set_epi64x((long long) g64[(W[3] >> 16 * i) & 0xFFFF], (long long) g64[(W[2] >> 16 * i) & 0xFFFF]);
            acc[h] = _mm256_dpbusd_avx_epi32(acc[h], _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1), Q[2 * i + h]);
        }
    }
    // lanes (j' lo, j' hi) per half: hadd -> j0 j1 j4 j5 | j2 j3 j6 j7 -> qword permute to j0..j7
    const __m256i si = _mm256_permute4x64_epi64(_mm256_hadd_epi32(acc[0], acc[1]), _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i QH = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) qhp));
    const __m256 sumi = _mm256_cvtepi32_ps(si);
    const __m256 sc = _mm256_cvtepi32_ps(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(QH, 11), _mm256_set1_epi32(0x0E)), _mm256_set1_epi32(1)));
    uint16_t dh; memcpy(&dh, b, 2);
    const __m256 dW = _mm256_mul_ps(_mm256_set1_ps(_cvtsh_ss(dh)), sc);                           // FMUL.FTZ
    const __m256 yd = _mm256_blendv_ps(_mm256_load_ps(q.dym + 8 * kb), _mm256_load_ps(q.dyp + 8 * kb), _mm256_castsi256_ps(_mm256_slli_epi32(QH, 16)));
    const __m256 inner = _mm256_fmadd_ps(_mm256_load_ps(q.dsx + 8 * kb), sumi, yd);              // FFMA.FTZ
    return _mm256_fmadd_ps(dW, inner, _mm256_setzero_ps());                                       // FFMA.FTZ dW*inner + 0
}

// IQ2_XXS block (66 B: half d, 8 x {u32 grid bytes, u32 signs|scale}) against q-block kb
__attribute__((always_inline)) static inline __m256 blk_iq2_xxs(const uint8_t * b, const QX & q, int kb) {
    const __m256 Y0 = _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i *) (b + 2)));
    const __m256 Y1 = _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i *) (b + 34)));
    const __m256i Q2 = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(Y0, Y1, _MM_SHUFFLE(2, 0, 2, 0))), _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i AX = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(Y0, Y1, _MM_SHUFFLE(3, 1, 3, 1))), _MM_SHUFFLE(3, 1, 2, 0));
    // byte k of lane j = 7-bit sign field k, then bit 7 = its parity: the 8 sign bits of grid group k
    __m256i B = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(AX, _mm256_set1_epi32(0x7F)), _mm256_and_si256(_mm256_slli_epi32(AX, 1), _mm256_set1_epi32(0x7F00))),
                                _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi32(AX, 2), _mm256_set1_epi32(0x7F0000)), _mm256_and_si256(_mm256_slli_epi32(AX, 3), _mm256_set1_epi32(0x7F000000))));
    const __m256i PT = _mm256_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0,
                                        0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0);
    const __m256i n4 = _mm256_set1_epi8(0x0F);
    B = _mm256_or_si256(B, _mm256_xor_si256(_mm256_shuffle_epi8(PT, _mm256_and_si256(B, n4)), _mm256_shuffle_epi8(PT, _mm256_and_si256(_mm256_srli_epi16(B, 4), n4))));
    const __m256i M1 = _mm256_set1_epi32(0x08040201), M2 = _mm256_set1_epi32((int) 0x80402010u), one = _mm256_set1_epi8(1);
    const int * G = (const int *) iq2xxs_grid;
    const __m256i * Q = Qreg(q, kb);
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
#define K3_IQ2_GROUP(k)                                                                                                  \
    {                                                                                                                    \
        const __m256i gi = (k) == 3 ? _mm256_srli_epi32(Q2, 24) : _mm256_and_si256(_mm256_srli_epi32(Q2, 8 * (k)), _mm256_set1_epi32(0xFF)); \
        const __m256i lo = _mm256_i32gather_epi32(G, gi, 8), hi = _mm256_i32gather_epi32(G + 1, gi, 8);                  \
        const __m256i S = _mm256_shuffle_epi8(B, _mm256_setr_epi32((k) * 0x01010101, (4 + (k)) * 0x01010101, (8 + (k)) * 0x01010101, (12 + (k)) * 0x01010101, \
                                                                   (k) * 0x01010101, (4 + (k)) * 0x01010101, (8 + (k)) * 0x01010101, (12 + (k)) * 0x01010101)); \
        const __m256i sl = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(S, M1), M1), one);                         \
        const __m256i sh = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(S, M2), M2), one);                         \
        a0 = _mm256_dpbusd_avx_epi32(a0, lo, _mm256_sign_epi8(Q[2 * (k)], sl));                                          \
        a1 = _mm256_dpbusd_avx_epi32(a1, hi, _mm256_sign_epi8(Q[2 * (k) + 1], sh));                                      \
    }
    K3_IQ2_GROUP(0) K3_IQ2_GROUP(1) K3_IQ2_GROUP(2) K3_IQ2_GROUP(3)
#undef K3_IQ2_GROUP
    __m256i s = _mm256_mullo_epi32(_mm256_add_epi32(a0, a1), _mm256_or_si256(_mm256_srli_epi32(AX, 27), _mm256_set1_epi32(1)));
    s = _mm256_srai_epi32(_mm256_add_epi32(s, _mm256_and_si256(_mm256_srai_epi32(s, 31), _mm256_set1_epi32(7))), 3);  // C int / 8
    uint16_t dh; memcpy(&dh, b, 2);
    const __m256 dW = _mm256_mul_ps(_mm256_set1_ps(_cvtsh_ss(dh)), _mm256_load_ps(q.dsx + 8 * kb));      // FMUL.FTZ
    return _mm256_fmadd_ps(dW, _mm256_cvtepi32_ps(s), _mm256_setzero_ps());
}

// IQ2_XXS with scalar loads (E-cores): per sub-block j and group k, the 8 magnitudes are iq2xxs_grid[byte k] (one u64)
// and the 8 sign multipliers (+1 / -1 bytes for vpsignb) ksg64[7-bit field k] (parity bit included), against q64.
alignas(64) static uint64_t ksg64[128];
static void init_ksg64() {
    for (int v = 0; v < 128; v++) {
        const int s8 = v | ((__builtin_popcount(v) & 1) << 7);
        uint64_t m = 0; for (int b = 0; b < 8; b++) m |= (uint64_t) ((s8 >> b) & 1 ? 0xFF : 0x01) << (8 * b);
        ksg64[v] = m;
    }
}
__attribute__((always_inline)) static inline __m256 blk_iq2_xxs_sc(const uint8_t * b, const QX & q, int kb) {
    const uint8_t * qs = b + 2;
    const __m256i * Q = (const __m256i *) (q.q64 + kb * 256);
    __m256i acc[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
#pragma GCC unroll 2
    for (int h = 0; h < 2; h++) {
        uint32_t q2[4], ax[4];
#pragma GCC unroll 4
        for (int jj = 0; jj < 4; jj++) { memcpy(&q2[jj], qs + 8 * (4 * h + jj), 4); memcpy(&ax[jj], qs + 8 * (4 * h + jj) + 4, 4); }
#pragma GCC unroll 4
        for (int k = 0; k < 4; k++) {
            const __m128i g01 = _mm_set_epi64x((long long) iq2xxs_grid[(q2[1] >> 8 * k) & 0xFF], (long long) iq2xxs_grid[(q2[0] >> 8 * k) & 0xFF]);
            const __m128i g23 = _mm_set_epi64x((long long) iq2xxs_grid[(q2[3] >> 8 * k) & 0xFF], (long long) iq2xxs_grid[(q2[2] >> 8 * k) & 0xFF]);
            const __m128i s01 = _mm_set_epi64x((long long) ksg64[(ax[1] >> 7 * k) & 0x7F], (long long) ksg64[(ax[0] >> 7 * k) & 0x7F]);
            const __m128i s23 = _mm_set_epi64x((long long) ksg64[(ax[3] >> 7 * k) & 0x7F], (long long) ksg64[(ax[2] >> 7 * k) & 0x7F]);
            const __m256i G = _mm256_inserti128_si256(_mm256_castsi128_si256(g01), g23, 1);
            const __m256i S = _mm256_inserti128_si256(_mm256_castsi128_si256(s01), s23, 1);
            acc[h] = _mm256_dpbusd_avx_epi32(acc[h], G, _mm256_sign_epi8(Q[2 * k + h], S));
        }
    }
    const __m256i si = _mm256_permute4x64_epi64(_mm256_hadd_epi32(acc[0], acc[1]), _MM_SHUFFLE(3, 1, 2, 0));
    const __m256 Y0 = _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i *) (b + 2)));
    const __m256 Y1 = _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i *) (b + 34)));
    const __m256i AX = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(Y0, Y1, _MM_SHUFFLE(3, 1, 3, 1))), _MM_SHUFFLE(3, 1, 2, 0));
    __m256i s = _mm256_mullo_epi32(si, _mm256_or_si256(_mm256_srli_epi32(AX, 27), _mm256_set1_epi32(1)));
    s = _mm256_srai_epi32(_mm256_add_epi32(s, _mm256_and_si256(_mm256_srai_epi32(s, 31), _mm256_set1_epi32(7))), 3);  // C int / 8
    uint16_t dh; memcpy(&dh, b, 2);
    const __m256 dW = _mm256_mul_ps(_mm256_set1_ps(_cvtsh_ss(dh)), _mm256_load_ps(q.dsx + 8 * kb));      // FMUL.FTZ
    return _mm256_fmadd_ps(dW, _mm256_cvtepi32_ps(s), _mm256_setzero_ps());
}

// Rows are done two at a time (gate i with up i, down r with r+1): the two share every activation load of a block.
// Per row the block results p[kb] go to a small stack array and are then reduced in CUDA's order: lanes
// l = 8*(kb%4) + j of warp kb/4, s_l = ((w0 + w1) + w2) + w3 (missing blocks are +0.0f), then the xor butterfly =
// tree [l]+[l+16], +[l+8], +[l+4], +[l+2], +[l+1]. (Fully unrolled with the sums in registers, GCC hoisted all 3.5 KB
// of activation loads to the top of the row and spilled them to the stack.)
template <int T, int SC> __attribute__((always_inline)) static inline __m256 blk(const uint8_t * row, const QX & q, int kb) {
    constexpr int BB = T == K3CPU_T_IQ1_S ? 50 : 66;
    return T == K3CPU_T_IQ1_S ? (SC ? blk_iq1_s_sc(row + BB * kb, q, kb) : blk_iq1_s(row + BB * kb, q, kb))
                              : (SC ? blk_iq2_xxs_sc(row + BB * kb, q, kb) : blk_iq2_xxs(row + BB * kb, q, kb));
}
template <int NB> __attribute__((always_inline)) static inline float reduce_row(const float * p) {
    const __m256 z = _mm256_setzero_ps();
    __m256 S[4];
    for (int m = 0; m < 4; m++)
        S[m] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_load_ps(p + 8 * m), _mm256_load_ps(p + 8 * (4 + m))), _mm256_load_ps(p + 8 * (8 + m))),
                             12 + m < NB ? _mm256_load_ps(p + 8 * (12 + m)) : z);
    const __m256 t = _mm256_add_ps(_mm256_add_ps(S[0], S[2]), _mm256_add_ps(S[1], S[3]));
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_shuffle_ps(x, x, 1));
    return _mm_cvtss_f32(x);
}
// rows ra and rb (same type T, NB blocks each, both against q) -> *oa, *ob
template <int T, int NB, int SC> __attribute__((noinline, optimize("no-tree-slp-vectorize"))) static void rows2(const uint8_t * ra, const uint8_t * rb, const QX & q, float * oa, float * ob) {
    alignas(32) float pa[NB * 8], pb[NB * 8];
#pragma GCC unroll 1
    for (int kb = 0; kb < NB; kb++) {
        _mm256_store_ps(pa + 8 * kb, blk<T, SC>(ra, q, kb));
        _mm256_store_ps(pb + 8 * kb, blk<T, SC>(rb, q, kb));
    }
    *oa = reduce_row<NB>(pa);
    *ob = reduce_row<NB>(pb);
}
template <int NB, int SC> static void rows2_t(int ta, int tb, const uint8_t * ra, const uint8_t * rb, const QX & q, float * oa, float * ob) {
    if (ta == tb) {
        if (ta == K3CPU_T_IQ1_S) rows2<K3CPU_T_IQ1_S, NB, SC>(ra, rb, q, oa, ob); else rows2<K3CPU_T_IQ2_XXS, NB, SC>(ra, rb, q, oa, ob);
        return;
    }
    float d;   // mixed types (not in Kimi K3): each row paired with itself
    if (ta == K3CPU_T_IQ1_S) rows2<K3CPU_T_IQ1_S, NB, SC>(ra, ra, q, oa, &d); else rows2<K3CPU_T_IQ2_XXS, NB, SC>(ra, ra, q, oa, &d);
    if (tb == K3CPU_T_IQ1_S) rows2<K3CPU_T_IQ1_S, NB, SC>(rb, rb, q, ob, &d); else rows2<K3CPU_T_IQ2_XXS, NB, SC>(rb, rb, q, ob, &d);
}
static thread_local int t_sc = 0;   // this worker's IQ1_S lookup: 0 = vpgatherdd, 1 = scalar loads
template <int NB> static inline void rows2_any(int ta, int tb, const uint8_t * ra, const uint8_t * rb, const QX & q, float * oa, float * ob) {
    if (t_sc) rows2_t<NB, 1>(ta, tb, ra, rb, q, oa, ob); else rows2_t<NB, 0>(ta, tb, ra, rb, q, oa, ob);
}

// ------------------------------------------------------------------------------------------------ v7 records
// k3x's GPU layout (../sycl/k3x.cpp v7_off): per group of 4 rows, block kb = 4i + q of row j has its dword u (block
// padded to 13 / 17 dwords) at  g + (i*ndw + u)*16 + 4q + j  for i < 3 and  g + 48*ndw + u*8 + 4q + j  for i = 3 (only
// kb 12, 13 of 14-block rows), g = rg4 * nblk*4*ndw. So one 16-byte load is dword u of the 4 rows: these kernels do
// 4 rows per pass with lanes (sub-block j = 2sp + h, row), and the CUDA reduction tree becomes lane-wise adds
// (rows4_v7). Bytes 2.. of a block (after the half d) straddle dwords, hence the 16-bit shifts.
template <int NDW> static inline const uint32_t * v7_blk(const uint32_t * grp, int kb, int & st) {
    const int i = kb >> 2, q = kb & 3;
    if (i < 3) { st = 16; return grp + (size_t) i * NDW * 16 + 4 * q; }
    st = 8; return grp + (size_t) 48 * NDW + 4 * q;
}
static inline __m256i ld2(const uint32_t * a, const uint32_t * b) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) a)), _mm_loadu_si128((const __m128i *) b), 1);
}
// d of the 4 rows (low half of dword 0), as [r0 r1 r2 r3 | r0 r1 r2 r3]
static inline __m256 v7_d4(const uint32_t * B) {
    const __m128i h = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) B), _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1));
    const __m128 f = _mm_cvtph_ps(h);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(f), f, 1);
}
// IQ1_S, block kb, sub-blocks 2sp and 2sp+1 of the 4 rows (B = dword 0 of row 0, st = dword stride)
template <int SC> __attribute__((always_inline)) static inline __m256 blk_iq1_s_v7(const uint32_t * B, int st, const QX & q, int kb, int sp, __m256 D4) {
    // qs dword j of a row = bytes 2+4j..5+4j = (D[j] >> 16) | (D[j+1] << 16); qh[j] = 16 bits at byte 34+2j
    const __m256i QS = _mm256_or_si256(_mm256_srli_epi32(ld2(B + 2 * sp * st, B + (2 * sp + 1) * st), 16),
                                       _mm256_slli_epi32(ld2(B + (2 * sp + 1) * st, B + (2 * sp + 2) * st), 16));
    const __m256i QH = _mm256_and_si256(_mm256_srlv_epi32(ld2(B + (8 + sp) * st, B + (9 + sp) * st), _mm256_setr_epi32(16, 16, 16, 16, 0, 0, 0, 0)), _mm256_set1_epi32(0xFFFF));
    __m256i si;
    const int l = kb * 4 + sp;
    if (!SC) {
        const __m256i m7 = _mm256_set1_epi32(0x700), mlo = _mm256_set1_epi32(0x0F0F0F0F), mhi = _mm256_set1_epi32((int) 0xF0F0F0F0u);
        const int * G = (const int *) iq1s_grid_gpu;
#define K3_SEL(i) _mm256_setr_epi8(i, -1, -1, -1, 4 + i, -1, -1, -1, 8 + i, -1, -1, -1, 12 + i, -1, -1, -1, i, -1, -1, -1, 4 + i, -1, -1, -1, 8 + i, -1, -1, -1, 12 + i, -1, -1, -1)
        const __m256i g0 = _mm256_i32gather_epi32(G, _mm256_or_si256(_mm256_shuffle_epi8(QS, K3_SEL(0)), _mm256_and_si256(_mm256_slli_epi32(QH, 8), m7)), 4);
        const __m256i g1 = _mm256_i32gather_epi32(G, _mm256_or_si256(_mm256_shuffle_epi8(QS, K3_SEL(1)), _mm256_and_si256(_mm256_slli_epi32(QH, 5), m7)), 4);
        const __m256i g2 = _mm256_i32gather_epi32(G, _mm256_or_si256(_mm256_shuffle_epi8(QS, K3_SEL(2)), _mm256_and_si256(_mm256_slli_epi32(QH, 2), m7)), 4);
        const __m256i g3 = _mm256_i32gather_epi32(G, _mm256_or_si256(_mm256_shuffle_epi8(QS, K3_SEL(3)), _mm256_and_si256(_mm256_srli_epi32(QH, 1), m7)), 4);
#undef K3_SEL
        const __m256i * Q = (const __m256i *) (q.qv + (size_t) l * 8 * 32);
        __m256i a0 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_and_si256(g0, mlo), Q[0]);
        __m256i a1 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_and_si256(g0, mhi), Q[1]);
        a0 = _mm256_dpbusd_avx_epi32(a0, _mm256_and_si256(g1, mlo), Q[2]);
        a1 = _mm256_dpbusd_avx_epi32(a1, _mm256_and_si256(g1, mhi), Q[3]);
        a0 = _mm256_dpbusd_avx_epi32(a0, _mm256_and_si256(g2, mlo), Q[4]);
        a1 = _mm256_dpbusd_avx_epi32(a1, _mm256_and_si256(g2, mhi), Q[5]);
        a0 = _mm256_dpbusd_avx_epi32(a0, _mm256_and_si256(g3, mlo), Q[6]);
        a1 = _mm256_dpbusd_avx_epi32(a1, _mm256_and_si256(g3, mhi), Q[7]);
        si = _mm256_add_epi32(a0, _mm256_srai_epi32(a1, 4));
    } else {
        // scalar lookups: lane (h, row) -> four 64-bit entries per register = the 4 rows of one sub-block. The 16-bit
        // fields are loaded straight from the v7 dwords (vector -> scalar through memory is slow on the E-cores).
        __m256i acc[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++) {
            const int j = 2 * sp + h;
            // activations: values 8i..8i+7 of q8_1 block 8kb+j, broadcast to the 4 rows (vpbroadcastq from q64)
            const int8_t * qj = q.q64 + kb * 256 + (j >> 2) * 32 + (j & 3) * 8;
            const uint16_t * w0 = (const uint16_t *) (B + j * st), * w1 = (const uint16_t *) (B + (j + 1) * st);
            const uint16_t * wh = (const uint16_t *) (B + ((34 + 2 * j) >> 2) * st) + (((34 + 2 * j) >> 1) & 1);
            uint64_t W[4];
#pragma GCC unroll 4
            for (int r = 0; r < 4; r++) {
                const uint32_t q4 = (uint32_t) w0[2 * r + 1] | (uint32_t) w1[2 * r] << 16;   // bytes 2+4j..5+4j of row r
                W[r] = _pdep_u64(q4, 0x00FF00FF00FF00FFull) | _pdep_u64(wh[2 * r], 0x0700070007000700ull);
            }
#pragma GCC unroll 4
            for (int i = 0; i < 4; i++) {
                const __m128i lo = _mm_set_epi64x((long long) g64[(W[1] >> 16 * i) & 0xFFFF], (long long) g64[(W[0] >> 16 * i) & 0xFFFF]);
                const __m128i hi = _mm_set_epi64x((long long) g64[(W[3] >> 16 * i) & 0xFFFF], (long long) g64[(W[2] >> 16 * i) & 0xFFFF]);
                int64_t qv8; memcpy(&qv8, qj + 64 * i, 8);
                acc[h] = _mm256_dpbusd_avx_epi32(acc[h], _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1), _mm256_set1_epi64x(qv8));
            }
        }
        // hadd: [h0 r0, h0 r1, h1 r0, h1 r1 | h0 r2, h0 r3, h1 r2, h1 r3] -> [h0 r0..r3 | h1 r0..r3]
        si = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(acc[0], acc[1]), _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    }
    const __m256 sumi = _mm256_cvtepi32_ps(si);
    const __m256 sc = _mm256_cvtepi32_ps(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(QH, 11), _mm256_set1_epi32(0x0E)), _mm256_set1_epi32(1)));
    const __m256 dW = _mm256_mul_ps(D4, sc);                                                       // FMUL.FTZ
    const __m256 yd = _mm256_blendv_ps(_mm256_load_ps(q.dym4 + 8 * l), _mm256_load_ps(q.dyp4 + 8 * l), _mm256_castsi256_ps(_mm256_slli_epi32(QH, 16)));
    const __m256 inner = _mm256_fmadd_ps(_mm256_load_ps(q.dsx4 + 8 * l), sumi, yd);               // FFMA.FTZ
    return _mm256_fmadd_ps(dW, inner, _mm256_setzero_ps());                                        // FFMA.FTZ dW*inner + 0
}
// IQ2_XXS, block kb, sub-blocks 2sp and 2sp+1 of the 4 rows: q2 = bytes 2+8j..5+8j, aux = bytes 6+8j..9+8j
template <int SC> __attribute__((always_inline)) static inline __m256 blk_iq2_xxs_v7(const uint32_t * B, int st, const QX & q, int kb, int sp, __m256 D4) {
    const uint32_t * b0 = B + 4 * sp * st;   // dword 2j of j = 2sp; j = 2sp+1 starts 2 dwords later
    const __m256i Y0 = ld2(b0, b0 + 2 * st), Y1 = ld2(b0 + st, b0 + 3 * st), Y2 = ld2(b0 + 2 * st, b0 + 4 * st);
    const __m256i Q2 = _mm256_or_si256(_mm256_srli_epi32(Y0, 16), _mm256_slli_epi32(Y1, 16));
    const __m256i AX = _mm256_or_si256(_mm256_srli_epi32(Y1, 16), _mm256_slli_epi32(Y2, 16));
    const int l = kb * 4 + sp;
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    if (!SC) {
        __m256i Bs = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(AX, _mm256_set1_epi32(0x7F)), _mm256_and_si256(_mm256_slli_epi32(AX, 1), _mm256_set1_epi32(0x7F00))),
                                     _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi32(AX, 2), _mm256_set1_epi32(0x7F0000)), _mm256_and_si256(_mm256_slli_epi32(AX, 3), _mm256_set1_epi32(0x7F000000))));
        const __m256i PT = _mm256_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0,
                                            0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0);
        const __m256i n4 = _mm256_set1_epi8(0x0F);
        Bs = _mm256_or_si256(Bs, _mm256_xor_si256(_mm256_shuffle_epi8(PT, _mm256_and_si256(Bs, n4)), _mm256_shuffle_epi8(PT, _mm256_and_si256(_mm256_srli_epi16(Bs, 4), n4))));
        const __m256i M1 = _mm256_set1_epi32(0x08040201), M2 = _mm256_set1_epi32((int) 0x80402010u), one = _mm256_set1_epi8(1);
        const int * G = (const int *) iq2xxs_grid;
        const __m256i * Q = (const __m256i *) (q.qv + (size_t) l * 8 * 32);
#define K3_IQ2_GROUP(k)                                                                                                  \
        {                                                                                                                \
            const __m256i gi = (k) == 3 ? _mm256_srli_epi32(Q2, 24) : _mm256_and_si256(_mm256_srli_epi32(Q2, 8 * (k)), _mm256_set1_epi32(0xFF)); \
            const __m256i lo = _mm256_i32gather_epi32(G, gi, 8), hi = _mm256_i32gather_epi32(G + 1, gi, 8);              \
            const __m256i S = _mm256_shuffle_epi8(Bs, _mm256_setr_epi32((k) * 0x01010101, (4 + (k)) * 0x01010101, (8 + (k)) * 0x01010101, (12 + (k)) * 0x01010101, \
                                                                        (k) * 0x01010101, (4 + (k)) * 0x01010101, (8 + (k)) * 0x01010101, (12 + (k)) * 0x01010101)); \
            const __m256i sl = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(S, M1), M1), one);                     \
            const __m256i sh = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(S, M2), M2), one);                     \
            a0 = _mm256_dpbusd_avx_epi32(a0, lo, _mm256_sign_epi8(Q[2 * (k)], sl));                                      \
            a1 = _mm256_dpbusd_avx_epi32(a1, hi, _mm256_sign_epi8(Q[2 * (k) + 1], sh));                                  \
        }
        K3_IQ2_GROUP(0) K3_IQ2_GROUP(1) K3_IQ2_GROUP(2) K3_IQ2_GROUP(3)
#undef K3_IQ2_GROUP
        a0 = _mm256_add_epi32(a0, a1);
    } else {
        __m256i acc[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++) {
            const int j = 2 * sp + h;
            const int8_t * qj = q.q64 + kb * 256 + (j >> 2) * 32 + (j & 3) * 8;
            // q2 = bytes 2+8j..5+8j = dwords 2j (high half), 2j+1 (low); aux = 2j+1 (high), 2j+2 (low)
            const uint16_t * w0 = (const uint16_t *) (B + 2 * j * st), * w1 = (const uint16_t *) (B + (2 * j + 1) * st), * w2 = (const uint16_t *) (B + (2 * j + 2) * st);
            uint32_t x2[4], xa[4];
#pragma GCC unroll 4
            for (int r = 0; r < 4; r++) { x2[r] = (uint32_t) w0[2 * r + 1] | (uint32_t) w1[2 * r] << 16; xa[r] = (uint32_t) w1[2 * r + 1] | (uint32_t) w2[2 * r] << 16; }
#pragma GCC unroll 4
            for (int k = 0; k < 4; k++) {
                const __m128i g01 = _mm_set_epi64x((long long) iq2xxs_grid[(x2[1] >> 8 * k) & 0xFF], (long long) iq2xxs_grid[(x2[0] >> 8 * k) & 0xFF]);
                const __m128i g23 = _mm_set_epi64x((long long) iq2xxs_grid[(x2[3] >> 8 * k) & 0xFF], (long long) iq2xxs_grid[(x2[2] >> 8 * k) & 0xFF]);
                const __m128i s01 = _mm_set_epi64x((long long) ksg64[(xa[1] >> 7 * k) & 0x7F], (long long) ksg64[(xa[0] >> 7 * k) & 0x7F]);
                const __m128i s23 = _mm_set_epi64x((long long) ksg64[(xa[3] >> 7 * k) & 0x7F], (long long) ksg64[(xa[2] >> 7 * k) & 0x7F]);
                int64_t qv8; memcpy(&qv8, qj + 64 * k, 8);
                acc[h] = _mm256_dpbusd_avx_epi32(acc[h], _mm256_inserti128_si256(_mm256_castsi128_si256(g01), g23, 1),
                                                 _mm256_sign_epi8(_mm256_set1_epi64x(qv8), _mm256_inserti128_si256(_mm256_castsi128_si256(s01), s23, 1)));
            }
        }
        a0 = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(acc[0], acc[1]), _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    }
    __m256i s = _mm256_mullo_epi32(a0, _mm256_or_si256(_mm256_srli_epi32(AX, 27), _mm256_set1_epi32(1)));
    s = _mm256_srai_epi32(_mm256_add_epi32(s, _mm256_and_si256(_mm256_srai_epi32(s, 31), _mm256_set1_epi32(7))), 3);  // C int / 8
    const __m256 dW = _mm256_mul_ps(D4, _mm256_load_ps(q.dsx4 + 8 * l));                         // FMUL.FTZ
    return _mm256_fmadd_ps(dW, _mm256_cvtepi32_ps(s), _mm256_setzero_ps());
}
// rows 4*rg4 .. 4*rg4+3 of a v7 matrix (NB blocks per row) -> out[0..3]
template <int T, int NB, int SC> __attribute__((noinline, optimize("no-tree-slp-vectorize")))
static void rows4_v7(const uint32_t * mat, int rg4, const QX & q, float * out) {
    constexpr int NDW = T == K3CPU_T_IQ1_S ? 13 : 17;
    const uint32_t * grp = mat + (size_t) rg4 * NB * 4 * NDW;
    __m256 D4[NB];
    for (int kb = 0; kb < NB; kb++) { int st; D4[kb] = v7_d4(v7_blk<NDW>(grp, kb, st)); }
    // block kb = m + 4w: w (= v7's i) is a compile-time constant, so the dword stride is too
#define K3_P(W, m, sp) (T == K3CPU_T_IQ1_S                                                                                     \
        ? blk_iq1_s_v7<SC>(grp + ((W) < 3 ? (W) * NDW * 16 : 48 * NDW) + 4 * (m), (W) < 3 ? 16 : 8, q, (m) + 4 * (W), sp, D4[(m) + 4 * (W)]) \
        : blk_iq2_xxs_v7<SC>(grp + ((W) < 3 ? (W) * NDW * 16 : 48 * NDW) + 4 * (m), (W) < 3 ? 16 : 8, q, (m) + 4 * (W), sp, D4[(m) + 4 * (W)]))
    const __m256 z = _mm256_setzero_ps();
    __m256 Tq[4];
#pragma GCC unroll 1
    for (int sp = 0; sp < 4; sp++) {
        __m256 S[4];
#pragma GCC unroll 1
        for (int m = 0; m < 4; m++) {   // lane l = 8m + j: ((w0 + w1) + w2) + w3 over kb = m, m+4, m+8, m+12
            __m256 a = K3_P(0, m, sp);
            a = _mm256_add_ps(a, K3_P(1, m, sp));
            a = _mm256_add_ps(a, K3_P(2, m, sp));
            a = _mm256_add_ps(a, m + 12 < NB ? K3_P(3, m, sp) : z);
            S[m] = a;
        }
        Tq[sp] = _mm256_add_ps(_mm256_add_ps(S[0], S[2]), _mm256_add_ps(S[1], S[3]));   // [l]+[l+16], then +[l+8]
    }
#undef K3_P
    const __m256 W = _mm256_add_ps(_mm256_add_ps(Tq[0], Tq[2]), _mm256_add_ps(Tq[1], Tq[3]));      // +[l+4], then +[l+2]
    _mm_storeu_ps(out, _mm_add_ps(_mm256_castps256_ps128(W), _mm256_extractf128_ps(W, 1)));       // +[l+1]
}
template <int NB> static void rows4_v7_any(int t, const uint32_t * mat, int rg4, const QX & q, float * out) {
    if (t == K3CPU_T_IQ1_S) { if (t_sc) rows4_v7<K3CPU_T_IQ1_S, NB, 1>(mat, rg4, q, out); else rows4_v7<K3CPU_T_IQ1_S, NB, 0>(mat, rg4, q, out); }
    else                    { if (t_sc) rows4_v7<K3CPU_T_IQ2_XXS, NB, 1>(mat, rg4, q, out); else rows4_v7<K3CPU_T_IQ2_XXS, NB, 0>(mat, rg4, q, out); }
}
static inline size_t v7_mat_bytes(int t, int nrows, int ncols) { return (size_t) nrows * (ncols / 256) * (t == K3CPU_T_IQ1_S ? 13 : 17) * 4; }

// ------------------------------------------------------------------------------------------------ table prefetch
// The packed MUFU tables (0.75 GB) miss in cache on almost every lookup; the SiTU chain is ex2 -> rcp (dependent), so
// the misses are prefetched a row ahead (tanh(g), tanh(u), ex2(g)) and the rcp one row later (once ex2 is cached).
// Index rules = k3x_mufu.h. Prefetch only: the values are computed by k3_act / nv_rcp unchanged.
static inline void pf_ent(const k3_ptab & P, uint32_t i) {
    if (i >= P.n) return;
    _mm_prefetch((const char *) (P.res + i), _MM_HINT_T0);
    _mm_prefetch((const char *) (P.hdr + 2 * (i >> 6)), _MM_HINT_T0);
}
static inline void pf_tanh(float x, const k3_mufu_tabs & M) {
    const uint32_t a = k3_f2u(k3_ftz(x)) & 0x7fffffffu;
    if (a >= 0x39800000u && a < 0x7f800000u) pf_ent(M.tanh, a - 0x39800000u);
}
static inline void pf_ex2(float x, const k3_mufu_tabs & M) {
    const uint32_t u = k3_f2u(k3_ftz(x)), a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return;
    if (!(u & 0x80000000u)) { if (a >= 0x33800000u) pf_ent(M.ex2p, a - 0x33800000u); }
    else if (a >= 0x33000000u) pf_ent(M.ex2n, a - 0x33000000u);
}
static inline void pf_rcp(float x, const k3_mufu_tabs & M) { pf_ent(M.rcp, k3_f2u(x) & 0x7fffffu); }
static inline void pf_act1(float g, float u, const k3_mufu_tabs & M) {   // the first-level lookups of k3_act
    pf_tanh(fmul_z(g, 0.25f), M); pf_tanh(fmul_z(u, k3_u2f(0x3D23D70Au)), M); pf_ex2(fmul_z(g, k3_u2f(0xBFB8AA3Bu)), M);
}
static inline void pf_act2(float g, const k3_mufu_tabs & M) {            // the dependent one: rcp(ex2(..) + 1)
    pf_rcp(fadd_z(nv_ex2(fmul_z(g, k3_u2f(0xBFB8AA3Bu)), M), 1.0f), M);
}

// ------------------------------------------------------------------------------------------------ MUFU tables
// Packed arrays live in 2 MiB transparent huge pages: lookups are random over 0.75 GB, and with 4 KiB pages nearly every
// one also missed the TLB (a page walk on top of the cache miss).
static void * huge_alloc(size_t n) {
    const size_t H = (size_t) 2 << 20, sz = (n + H - 1) / H * H;
    uint8_t * m = (uint8_t *) mmap(nullptr, sz + H, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    uint8_t * a = (uint8_t *) (((uintptr_t) m + H - 1) & ~(uintptr_t) (H - 1));
    if (a > m) munmap(m, a - m);
    if (m + sz + H > a + sz) munmap(a + sz, m + sz + H - (a + sz));
    madvise(a, sz, MADV_HUGEPAGE);
    return a;
}
struct Packed { uint32_t * hdr = nullptr; int8_t * res = nullptr; std::vector<uint32_t> esc; uint32_t n = 0, nb = 0; };
static Packed g_packed[4];

// k3x's pack_table (blocks of 64: {a, d} + int8 residuals + escapes), parallel over block ranges, verified per entry
static bool pack_table(const uint32_t * t, uint32_t n, Packed & P) {
    const uint32_t nb = (n + 63) / 64;
    P.n = n; P.nb = nb; P.esc.clear();
    P.hdr = (uint32_t *) huge_alloc(8 * (size_t) nb); P.res = (int8_t *) huge_alloc(n);
    if (!P.hdr || !P.res) { fprintf(stderr, "k3cpu: table allocation failed\n"); return false; }
    const int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::vector<uint32_t>> esc(nt);
    std::vector<std::thread> th;
    for (int ti = 0; ti < nt; ti++) th.emplace_back([&, ti] {
        const uint32_t b0 = (uint32_t) ((uint64_t) nb * ti / nt), b1 = (uint32_t) ((uint64_t) nb * (ti + 1) / nt);
        for (uint32_t b = b0; b < b1; b++) {
            const uint32_t i0 = b * 64, len = std::min(64u, n - i0);
            const uint32_t a = t[i0];
            int64_t dd = len == 64 ? (int64_t) t[i0 + 63] - (int64_t) a : 0;
            if (dd > (1 << 24) || dd < -(1 << 24)) dd = 0;
            const int32_t d = (int32_t) dd;
            P.hdr[2 * b] = a; P.hdr[2 * b + 1] = (uint32_t) d;
            for (uint32_t o = 0; o < len; o++) {
                const uint32_t ap = a + (uint32_t) ((d * (int32_t) o) / 63);
                const int64_t r = (int64_t) (int32_t) (t[i0 + o] - ap);
                if (r >= -127 && r <= 127) P.res[i0 + o] = (int8_t) r;
                else { P.res[i0 + o] = -128; esc[ti].push_back(i0 + o); esc[ti].push_back(t[i0 + o]); }
            }
        }
    });
    for (auto & x : th) x.join();
    for (auto & e : esc) P.esc.insert(P.esc.end(), e.begin(), e.end());
    const uint32_t nesc = (uint32_t) (P.esc.size() / 2);
    if (P.esc.empty()) P.esc.assign(2, 0);   // never read (no -128 residual), keeps the pointer valid
    const k3_ptab v{P.hdr, P.res, P.esc.data(), nesc, n};
    std::atomic<uint32_t> bad{0};
    th.clear();
    for (int ti = 0; ti < nt; ti++) th.emplace_back([&, ti] {
        const uint32_t i0 = (uint32_t) ((uint64_t) n * ti / nt), i1 = (uint32_t) ((uint64_t) n * (ti + 1) / nt);
        for (uint32_t i = i0; i < i1; i++) if (k3_ptab_get(v, i) != t[i]) { bad++; break; }
    });
    for (auto & x : th) x.join();
    return bad == 0;
}

bool k3cpu_load_mufu(const char * dir, k3_mufu_tabs * out) {
    const char * names[4] = {"rcp", "ex2p", "ex2n", "tanh"};
    k3_ptab tp[4] = {};
    for (int t = 0; t < 4; t++) {
        const std::string p = std::string(dir) + "/" + names[t] + ".bin";
        const int fd = open(p.c_str(), O_RDONLY);
        if (fd < 0) { perror(p.c_str()); return false; }
        struct stat st; fstat(fd, &st);
        void * m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (m == MAP_FAILED) { perror("mmap"); return false; }
        madvise(m, st.st_size, MADV_SEQUENTIAL);
        madvise(m, st.st_size, MADV_WILLNEED);
        Packed & P = g_packed[t];
        const bool ok = pack_table((const uint32_t *) m, (uint32_t) (st.st_size / 4), P);
        munmap(m, st.st_size);
        if (!ok) { fprintf(stderr, "k3cpu: pack verify FAILED for %s\n", p.c_str()); return false; }
        uint32_t nesc = 0; for (uint32_t i = 0; i < P.n; i++) nesc += P.res[i] == -128;
        tp[t] = k3_ptab{P.hdr, P.res, P.esc.data(), nesc, P.n};
    }
    *out = k3_mufu_tabs{tp[0], tp[1], tp[2], tp[3], tp[1].n, tp[2].n, tp[3].n};
    size_t bytes = 0; for (auto & P : g_packed) bytes += (size_t) P.nb * 8 + P.n + P.esc.size() * 4;
    fprintf(stderr, "k3cpu: MUFU tables packed, %.2f GB\n", bytes / 1e9);
    return true;
}

// ------------------------------------------------------------------------------------------------ pool
struct alignas(64) PaddedCounter { std::atomic<int> v{0}; char pad[60]; };

struct ExpertJob {
    k3cpu_expert e;
    const uint8_t * gate, * up, * down;
    size_t rbg, rbu, rbd;
    alignas(64) float h[F];
    QX qh;   // q8_1(h) in the dot layout (12 q-blocks used)
};

struct k3cpu_pool {
    k3_mufu_tabs mt;
    k3cpu_opts o;
    std::vector<int> cpus;
    std::vector<std::thread> th;
    alignas(64) std::atomic<uint32_t> gen{0};
    alignas(64) std::atomic<uint64_t> claim{0};   // (gen << 32) | next task
    alignas(64) std::atomic<int> sleepers{0};
    std::atomic<bool> stop{false};
    // job
    std::atomic<int> ntask{0};
    int k = 0;
    bool interleave = false;
    uint64_t t0 = 0;
    QX qx;
    ExpertJob job[K3CPU_KMAX];
    PaddedCounter gu_done[K3CPU_KMAX], dn_done[K3CPU_KMAX];
    std::atomic<uint64_t> t_gu[K3CPU_KMAX], t_dn[K3CPU_KMAX];
    uint64_t t_x = 0;
    const float * x = nullptr;
    alignas(64) std::atomic<int> qx_done{0};
    int sc_of[256] = {}, ecore_of[256] = {};
    int spin_mode = 0; const uint8_t * warm = nullptr;
};

static long futex(std::atomic<uint32_t> * a, int op, uint32_t v) { return syscall(SYS_futex, (uint32_t *) a, op, v, nullptr, nullptr, 0); }

static int g_pf_rows = 0;   // K3CPU_PF_ROWS: software prefetch distance in rows (0 = hardware prefetch only)
static inline void pf_bytes(const uint8_t * p, size_t n) { for (size_t o = 0; o < n; o += 64) _mm_prefetch((const char *) p + o, _MM_HINT_T0); }
static void do_gu(k3cpu_pool * P, int ei, int c) {
    ExpertJob & J = P->job[ei];
    const k3_mufu_tabs & M = P->mt;
    float g[CH_GU], u[CH_GU];
    const int i0 = c * CH_GU;
    const int pfd = g_pf_rows;
    const bool v7 = J.e.layout == K3CPU_V7;
    const int U = v7 ? 4 : 1;
    for (int i = 0; i < CH_GU; i++) {
        if (v7) {
            if (!(i & 3)) {
                rows4_v7_any<NB_X>(J.e.tg, (const uint32_t *) J.gate, (i0 + i) / 4, P->qx, &g[i]);
                rows4_v7_any<NB_X>(J.e.tu, (const uint32_t *) J.up, (i0 + i) / 4, P->qx, &u[i]);
            }
        } else {
            if (pfd && i + pfd < CH_GU) { pf_bytes(J.gate + (size_t) (i0 + i + pfd) * J.rbg, J.rbg); pf_bytes(J.up + (size_t) (i0 + i + pfd) * J.rbu, J.rbu); }
            rows2_any<NB_X>(J.e.tg, J.e.tu, J.gate + (size_t) (i0 + i) * J.rbg, J.up + (size_t) (i0 + i) * J.rbu, P->qx, &g[i], &u[i]);
        }
        // SiTU pipeline, U = rows produced per step (1, or 4 for v7): first-level table lines for row i now, the
        // dependent rcp line U rows later (once ex2 is cached), the activation itself 2U rows later
        pf_act1(g[i], u[i], M);
        if (i >= U) pf_act2(g[i - U], M);
        if (i >= 2 * U) J.h[i0 + i - 2 * U] = k3_act(g[i - 2 * U], u[i - 2 * U], M);
    }
    for (int i = CH_GU - U; i < CH_GU; i++) pf_act2(g[i], M);
    for (int i = CH_GU - 2 * U; i < CH_GU; i++) J.h[i0 + i] = k3_act(g[i], u[i], M);
    q8_block(J.h + i0, J.qh, c, M);
    if (J.e.g) memcpy(J.e.g + i0, g, sizeof g);
    if (J.e.u) memcpy(J.e.u + i0, u, sizeof u);
    if (J.e.h) memcpy(J.e.h + i0, J.h + i0, sizeof g);
    if (P->gu_done[ei].v.fetch_add(1, std::memory_order_acq_rel) + 1 == NGU) P->t_gu[ei].store(now_ns() - P->t0, std::memory_order_release);
}

static void do_dn(k3cpu_pool * P, int ei, int c) {
    ExpertJob & J = P->job[ei];
    while (P->gu_done[ei].v.load(std::memory_order_acquire) < NGU) _mm_pause();
    const int r0 = c * CH_DN;
    float * out = J.e.out;
    if (J.e.layout == K3CPU_V7) {
        for (int r = 0; r < CH_DN; r += 4) rows4_v7_any<NB_H>(J.e.td, (const uint32_t *) J.down, (r0 + r) / 4, J.qh, &out[r0 + r]);
    } else {
        const int pfd = g_pf_rows;
        for (int r = 0; r < CH_DN; r += 2) {
            if (pfd && r + 2 * pfd < CH_DN) pf_bytes(J.down + (size_t) (r0 + r + 2 * pfd) * J.rbd, 2 * J.rbd);
            rows2_any<NB_H>(J.e.td, J.e.td, J.down + (size_t) (r0 + r) * J.rbd, J.down + (size_t) (r0 + r + 1) * J.rbd, J.qh, &out[r0 + r], &out[r0 + r + 1]);
        }
    }
    if (P->dn_done[ei].v.fetch_add(1, std::memory_order_acq_rel) + 1 == NDN) P->t_dn[ei].store(now_ns() - P->t0, std::memory_order_release);
}

// q8_1 of x blocks [b0, b0 + QX_BLK): each block's rcp(d) line is prefetched first, so the misses overlap
static void do_qx(k3cpu_pool * P, int c) {
    const float * x = P->x;
    const int b0 = c * QX_BLK;
    for (int b = b0; b < b0 + QX_BLK; b++) {
        __m256 A = _mm256_setzero_ps();
        for (int i = 0; i < 32; i += 8) { const __m256 v = _mm256_loadu_ps(x + 32 * b + i); A = _mm256_max_ps(A, _mm256_max_ps(v, _mm256_sub_ps(_mm256_setzero_ps(), v))); }
        __m128 a4 = _mm_max_ps(_mm256_castps256_ps128(A), _mm256_extractf128_ps(A, 1));
        a4 = _mm_max_ps(a4, _mm_movehl_ps(a4, a4)); a4 = _mm_max_ss(a4, _mm_shuffle_ps(a4, a4, 1));
        pf_rcp(_mm_cvtss_f32(a4) * k3_u2f(0x3C010204u), P->mt);   // a hint only: exactness does not depend on it
    }
    for (int b = b0; b < b0 + QX_BLK; b++) q8_block(x + 32 * b, P->qx, b, P->mt);
    P->qx_done.fetch_add(1, std::memory_order_acq_rel);
}

static void run_tasks(k3cpu_pool * P, uint32_t my) {
    for (;;) {
        uint64_t v = P->claim.load(std::memory_order_acquire);
        if ((uint32_t) (v >> 32) != my) return;
        int t = (int) (uint32_t) v;
        if (t >= P->ntask.load(std::memory_order_relaxed)) return;
        if (!P->claim.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel)) continue;
        if (t < NQX) { do_qx(P, t); continue; }
        t -= NQX;
        while (P->qx_done.load(std::memory_order_acquire) < NQX) _mm_pause();
        if (P->interleave) {
            const int e = t / (NGU + NDN), r = t % (NGU + NDN);
            if (r < NGU) do_gu(P, e, r); else do_dn(P, e, r - NGU);
        } else {
            const int ngu = P->k * NGU;
            if (t < ngu) do_gu(P, t / NGU, t % NGU); else do_dn(P, (t - ngu) / NDN, (t - ngu) % NDN);
        }
    }
}

static void worker(k3cpu_pool * P, int idx) {
    if (!P->cpus.empty()) {
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(P->cpus[idx], &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    if (P->o.rt_prio > 0) {
        sched_param sp{}; sp.sched_priority = P->o.rt_prio;
        const int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
        if (rc && idx == 0) fprintf(stderr, "k3cpu: SCHED_FIFO %d not allowed (%s), workers stay SCHED_OTHER\n", P->o.rt_prio, strerror(rc));
    }
    set_daz_ftz();
    // CPUID 0x1A: core type of the cpu this thread now runs on (0x20 = Atom/E-core, 0x40 = Core/P-core)
    unsigned a = 0, b_ = 0, c = 0, d = 0;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b_), "=c"(c), "=d"(d) : "a"(0x1A), "c"(0));
    const bool ecore = (a >> 24) == 0x20;
    t_sc = P->o.lookup == 1 ? 0 : P->o.lookup == 2 ? 1 : ecore ? 1 : 0;
    P->sc_of[idx] = t_sc; P->ecore_of[idx] = ecore;
    uint32_t my = 0;
    for (;;) {
        uint32_t g = P->gen.load(std::memory_order_acquire);
        if (g == my) {
            const uint64_t t0 = now_ns();
            const uint64_t lim = P->o.spin_us < 0 ? UINT64_MAX : (uint64_t) P->o.spin_us * 1000;
            int n = 0;
            const int mode = P->spin_mode;
            size_t woff = 0;
            while ((g = P->gen.load(std::memory_order_acquire)) == my) {
                if (mode != 1) _mm_pause();
                if (++n == 256) {
                    n = 0; if (now_ns() - t0 > lim) break;
                    if (mode == 2 && P->warm) {   // keep the memory path busy: one line from a 64 MiB buffer per 256 spins
                        woff = (woff + 4096 + 64 * idx) & ((64u << 20) - 1);
                        (void) *(volatile const uint8_t *) (P->warm + woff);
                    }
                }
            }
            while ((g = P->gen.load(std::memory_order_acquire)) == my) {
                P->sleepers.fetch_add(1);
                if (P->gen.load() == my) futex(&P->gen, FUTEX_WAIT_PRIVATE, my);
                P->sleepers.fetch_sub(1);
            }
        }
        if (P->stop.load()) return;
        my = g;
        run_tasks(P, my);
    }
}

static bool check_f16c() {   // _cvtsh_ss (VCVTPH2PS) under DAZ|FTZ vs the exact conversion, all 65536 halves
    for (uint32_t h = 0; h < 65536; h++) {
        const float a = _cvtsh_ss((uint16_t) h), b = k3_h2f((uint16_t) h);
        if (k3_f2u(a) != k3_f2u(b) && !(a != a && b != b)) { fprintf(stderr, "k3cpu: F16C mismatch at half 0x%04x\n", h); return false; }
    }
    return true;
}

k3cpu_pool * k3cpu_create(const k3_mufu_tabs * mt, const k3cpu_opts & o) {
    const unsigned csr = _mm_getcsr(); set_daz_ftz();
    const bool ok = check_f16c();
    _mm_setcsr(csr);
    if (!ok) return nullptr;
    init_g64(); init_ksg64();
    if (getenv("K3CPU_PF_ROWS")) g_pf_rows = atoi(getenv("K3CPU_PF_ROWS"));
    k3cpu_pool * P = new k3cpu_pool();
    P->mt = *mt; P->o = o;
    if (getenv("K3CPU_SPIN_MODE")) P->spin_mode = atoi(getenv("K3CPU_SPIN_MODE"));
    if (P->spin_mode == 2) { uint8_t * w = (uint8_t *) huge_alloc(64u << 20); if (w) { memset(w, 1, 64u << 20); P->warm = w; } }
    const int n = o.ncpu > 0 ? o.ncpu : 1;
    if (o.cpus) P->cpus.assign(o.cpus, o.cpus + n);
    for (int i = 0; i < n; i++) P->th.emplace_back(worker, P, i);
    return P;
}

void k3cpu_destroy(k3cpu_pool * P) {
    if (!P) return;
    P->stop.store(true);
    P->gen.fetch_add(1);
    futex(&P->gen, FUTEX_WAKE_PRIVATE, INT_MAX);
    for (auto & t : P->th) t.join();
    delete P;
}

void k3cpu_submit(k3cpu_pool * P, const float * x, const k3cpu_expert * e, int k) {
    const uint64_t t0 = now_ns();
    k3cpu_wait_all(P);   // the previous job (if any) must be complete
    P->x = x;
    auto rb = [](int t, int kk) { return (size_t) (kk / 256) * (t == K3CPU_T_IQ1_S ? 50 : 66); };
    bool any_v7 = false;
    for (int i = 0; i < k; i++) {
        ExpertJob & J = P->job[i];
        J.e = e[i];
        J.rbg = rb(e[i].tg, D); J.rbu = rb(e[i].tu, D); J.rbd = rb(e[i].td, F);
        const bool v7 = e[i].layout == K3CPU_V7;
        J.gate = e[i].rec;
        J.up = J.gate + (v7 ? v7_mat_bytes(e[i].tg, F, D) : J.rbg * F);
        J.down = J.up + (v7 ? v7_mat_bytes(e[i].tu, F, D) : J.rbu * F);
        J.qh.v7 = v7; any_v7 |= v7;
        P->gu_done[i].v.store(0, std::memory_order_relaxed);
        P->dn_done[i].v.store(0, std::memory_order_relaxed);
        P->t_gu[i].store(0, std::memory_order_relaxed); P->t_dn[i].store(0, std::memory_order_relaxed);
    }
    P->qx.v7 = any_v7;
    P->k = k;
    P->interleave = P->o.interleave;
    P->t0 = t0;
    P->qx_done.store(0, std::memory_order_relaxed);
    P->ntask.store(NQX + k * (NGU + NDN), std::memory_order_relaxed);
    const uint32_t g = P->gen.load() + 1;
    P->claim.store((uint64_t) g << 32, std::memory_order_release);
    P->gen.store(g, std::memory_order_seq_cst);
    if (P->sleepers.load(std::memory_order_seq_cst) > 0) futex(&P->gen, FUTEX_WAKE_PRIVATE, INT_MAX);
    P->t_x = now_ns() - t0;
}

bool k3cpu_done(k3cpu_pool * P, int e) { return P->dn_done[e].v.load(std::memory_order_acquire) >= NDN; }
void k3cpu_wait(k3cpu_pool * P, int e) { while (P->dn_done[e].v.load(std::memory_order_acquire) < NDN) _mm_pause(); }
void k3cpu_wait_all(k3cpu_pool * P) { for (int i = 0; i < P->k; i++) k3cpu_wait(P, i); }
void k3cpu_run(k3cpu_pool * P, const float * x, const k3cpu_expert * e, int k) { k3cpu_submit(P, x, e, k); k3cpu_wait_all(P); }
void k3cpu_last_times(k3cpu_pool * P, k3cpu_times * t) {   // the last chunk's thread stores the time just after its count
    memset(t, 0, sizeof *t);
    t->x_done = P->t_x;
    for (int i = 0; i < P->k; i++) {
        while (!P->t_dn[i].load(std::memory_order_acquire) || !P->t_gu[i].load(std::memory_order_acquire)) _mm_pause();
        t->gu_done[i] = P->t_gu[i].load(); t->dn_done[i] = P->t_dn[i].load();
    }
}
