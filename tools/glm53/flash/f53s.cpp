// f53s: CUDA-exact CPU server for GLM-5.3-Flash routed experts (the NUC side of the batched remote expert tier,
// window nd, 2026-11-07). One request carries every (expert, token) pair a box owns in one MoE layer of one decode
// step, for up to TMAX tokens (sequences), so a decode step costs one round trip per layer for all streams.
//
// Arithmetic: the production GB10 path for Flash's regular expert layers (gate/up IQ2_XS 4096x2048, down IQ3_XXS
// 2048x4096): quantize_q8_1(x); mmvq ncols_dst=1 rows (4 warps, 1 row per block, vdr 2: g53exact::row's thread
// order); fused GLU swiglu_clamp(limit 10): gate=fminf(gate,L), up=fmaxf(fminf(up,L),-L), silu(gate)*up with MUFU
// ex2/rcp; quantize_q8_1(h); mmvq down rows. The AVX2/VNNI row is tools/glm53/cpu/g53cpu.cpp's fast_row (the same
// FMA order and FTZ/NaN canonicalisation), generalised to K=4096. Checked against GPU rows with `f53s check`.
//
// Wire (little endian): request = int32 hdr[4 + 2*KMAX] { magic|k, ntok, seq, 0, (gid, tok) x k } followed by
// ntok * 4096 f32 inputs; reply = k rows of 4096 f32 in request order. gid = engine layer * 288 + expert. A pair the
// box does not hold gets a NaN row (the engine computes it itself). The magic names the row order the engine's own
// kernel uses, so a box can never answer in the other one: 0x4E440000 = the four-warp ncols_dst=1 kernel
// (GGML_CUDA_MMVQ_EXACT_COLS=1, windows nd0-nd9), 0x4E450000 = the one-warp-per-token MoE kernel
// (GGML_CUDA_BATCH_INVARIANT=1, window nuc5). A box built before the second magic drops the connection on it.
//
// usage: f53s serve <records.bin> <records.idx> <port>     (idx lines: "gid offset"; records gate|up|down, 8060928 B)
//        f53s check <recs8.bin> <flashrow.bin>              (8 records, the flashrow cases)
//        f53s bench <recs8.bin> <k> <ntok> <reps>
// env: F53_MUFU=<dir with rcp/ex2p/ex2n.bin> (required), F53_CPUS=0,1,... (worker cpus), F53_SPIN_US (idle spin, 2000),
//      F53_ORDER=moe (check and bench only: rows in the one-warp MoE kernel's order; serve takes it from the magic)
#include "../g53exact.h"
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_SYCL
#include "../ref/ggml-common.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cstring>
#include <string>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <immintrin.h>
#include <linux/futex.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>
using namespace g53exact;

static constexpr int D = 4096, F = 2048, NBG = D / 256, NBD = F / 256;
static constexpr int KMAX = 32, TMAX = 8, NEXP = 288, NLAYER = 48;
static constexpr size_t GSZ = size_t(F) * NBG * 74, USZ = GSZ, DSZ = size_t(D) * NBD * 98, REC = GSZ + USZ + DSZ;
static constexpr uint32_t MAGIC = 0x4E440000u, MAGIC_MOE = 0x4E450000u;
static constexpr float LIMIT = 10.0f;

static int sum4(__m128i v) {
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    return _mm_cvtsi128_si32(v);
}
static __m256i dp(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), a, b); }
struct alignas(64) RowInput {
    __m256i q[D / 32];
    int correction[D / 32][2];
    alignas(32) int corr_lo[D / 32];   // the same corrections, one vector of 8 per 256-value block (fast_row2)
    alignas(32) int corr_hi[D / 32];
    __m256 scale[D / 256];
};
static void prepare(RowInput & dst, const Q8 * src, int nb) {
    for (int b = 0; b < nb; ++b) {
        alignas(16) uint16_t hd[8];
        for (int j = 0; j < 8; ++j) {
            const int n = b * 8 + j;
            dst.q[n] = _mm256_loadu_si256((const __m256i *) src[n].q);
            int lo = 0, hi = 0;
            for (int i = 0; i < 16; ++i) { lo += src[n].q[i]; hi += src[n].q[i + 16]; }
            dst.correction[n][0] = lo * 128;
            dst.correction[n][1] = hi * 128;
            dst.corr_lo[n] = lo * 128;
            dst.corr_hi[n] = hi * 128;
            hd[j] = src[n].d;
        }
        dst.scale[b] = _mm256_cvtph_ps(_mm_load_si128((const __m128i *) hd));
    }
}
struct IntDot { int a, c, sc; };
template <int ty> static inline IntDot isum(const uint8_t * b, int j, const RowInput & y, int n) {
    __m256i w;
    unsigned sc = 0;
    if (ty == 17) {
        const uint8_t * q = b + 2 + 8 * j;
        unsigned a = u16(q), c = u16(q + 2), d = u16(q + 4), e = u16(q + 6);
        w = _mm256_setr_epi64x(iq2xs_grid[a & 511], iq2xs_grid[c & 511], iq2xs_grid[d & 511], iq2xs_grid[e & 511]);
        auto s = _mm256_setr_epi64x(ksigns64[a >> 9], ksigns64[c >> 9], ksigns64[d >> 9], ksigns64[e >> 9]);
        w = _mm256_sign_epi8(w, _mm256_or_si256(s, _mm256_set1_epi8(1)));
        sc = b[66 + j];
    } else {
        const uint8_t * q = b + 2 + 8 * j;
        sc = u32(b + 66 + 4 * j);
        w = _mm256_setr_epi32(iq3xxs_grid[q[0]], iq3xxs_grid[q[1]], iq3xxs_grid[q[2]], iq3xxs_grid[q[3]],
                              iq3xxs_grid[q[4]], iq3xxs_grid[q[5]], iq3xxs_grid[q[6]], iq3xxs_grid[q[7]]);
        auto s = _mm256_setr_epi64x(ksigns64[sc & 127], ksigns64[(sc >> 7) & 127], ksigns64[(sc >> 14) & 127],
                                    ksigns64[(sc >> 21) & 127]);
        w = _mm256_sign_epi8(w, _mm256_or_si256(s, _mm256_set1_epi8(1)));
    }
    auto yy = y.q[n];
    w = _mm256_xor_si256(w, _mm256_set1_epi8(char(128)));
    auto v = dp(w, yy);
    int a = sum4(_mm256_castsi256_si128(v)) - y.correction[n][0], c = sum4(_mm256_extracti128_si256(v, 1)) - y.correction[n][1];
    return { a, c, int(sc) };
}
static __m256 clean(__m256 x) {
    auto u = _mm256_castps_si256(x), a = _mm256_and_si256(u, _mm256_set1_epi32(0x7fffffff));
    auto sub = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800000), a);
    u = _mm256_blendv_epi8(u, _mm256_and_si256(u, _mm256_set1_epi32(int(0x80000000))), sub);
    return _mm256_castsi256_ps(_mm256_blendv_epi8(u, _mm256_set1_epi32(0x7fffffff), _mm256_cmpgt_epi32(a, _mm256_set1_epi32(0x7f800000))));
}
// p[b % 16] packs virtual CUDA threads 8*(b%16) .. +7 (kbx = tid/8 + 16*i): each thread's increasing-kb FMA sequence.
// ng = 16: the four-warp ncols_dst=1 kernel (mul_mat_vec_q). ng = 4: the one-warp-per-token MoE kernel
// (mul_mat_vec_q_moe: 32 threads, kbx = tid/8 + 4*i, warp_reduce_sum), which GGML_CUDA_BATCH_INVARIANT=1 uses for
// every MUL_MAT_ID width including one.
template <int ng> static inline __m256 reduce_groups(const __m256 * p) {
    __m256 lane[4];
    if constexpr (ng == 16) {
        for (int i = 0; i < 4; i++)
            lane[i] = clean(_mm256_add_ps(clean(_mm256_add_ps(clean(_mm256_add_ps(p[i], p[i + 4])), p[i + 8])), p[i + 12]));
    } else {
        for (int i = 0; i < 4; i++) lane[i] = p[i];
    }
    lane[0] = clean(_mm256_add_ps(lane[0], lane[2]));   // shfl xor 16
    lane[1] = clean(_mm256_add_ps(lane[1], lane[3]));
    lane[0] = clean(_mm256_add_ps(lane[0], lane[1]));   // xor 8
    auto v = clean(_mm256_add_ps(lane[0], _mm256_permute2f128_ps(lane[0], lane[0], 1))); // xor 4
    v = clean(_mm256_add_ps(v, _mm256_permute_ps(v, 0x4e)));                              // xor 2
    return clean(_mm256_add_ps(v, _mm256_permute_ps(v, 0xb1)));                           // xor 1
}
template <int ty, int nb, int ng = 16> static float fast_row(const uint8_t * r, const RowInput * y) {
    __m256 p[ng];
    for (auto & v : p) v = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        alignas(32) int aa[8], cc[8], ss[8];
        const auto * w = r + b * blockbytes(ty);
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) {
            auto dot = isum<ty>(w, j, *y, b * 8 + j);
            if constexpr (ty == 18) {
                int total = dot.a + dot.c;
                ss[j] = (int(unsigned(dot.sc) >> 28) * total + total / 2) / 2;
            } else { aa[j] = dot.a; cc[j] = dot.c; ss[j] = dot.sc; }
        }
        __m256i sum;
        if constexpr (ty == 18) {
            sum = _mm256_load_si256((const __m256i *) ss);
        } else {
            auto a = _mm256_load_si256((const __m256i *) aa), c = _mm256_load_si256((const __m256i *) cc), sc = _mm256_load_si256((const __m256i *) ss);
            auto total = _mm256_add_epi32(a, c);
            auto halfsum = _mm256_srai_epi32(_mm256_add_epi32(total, _mm256_srli_epi32(total, 31)), 1);
            sum = _mm256_add_epi32(halfsum, _mm256_add_epi32(_mm256_mullo_epi32(a, _mm256_and_si256(sc, _mm256_set1_epi32(15))),
                                                             _mm256_mullo_epi32(c, _mm256_srli_epi32(sc, 4))));
            auto bias = _mm256_and_si256(_mm256_srai_epi32(sum, 31), _mm256_set1_epi32(3));
            sum = _mm256_srai_epi32(_mm256_add_epi32(sum, bias), 2);
        }
        auto d = clean(_mm256_mul_ps(_mm256_set1_ps(half(u16(w))), y->scale[b]));
        p[b % ng] = clean(_mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sum), p[b % ng]));
    }
    return _mm_cvtss_f32(_mm256_castps256_ps128(reduce_groups<ng>(p)));
}
// The same row with the 16 per-j horizontal sums replaced by one transposed hadd reduction per block (integer adds
// in another order: exact). F53_ROW=1 selects fast_row (the original) for comparison.
template <int ty> static inline __m256i wdec(const uint8_t * b, int j) {
    __m256i w;
    if (ty == 17) {
        const uint8_t * q = b + 2 + 8 * j;
        unsigned a = u16(q), c = u16(q + 2), d = u16(q + 4), e = u16(q + 6);
        w = _mm256_setr_epi64x(iq2xs_grid[a & 511], iq2xs_grid[c & 511], iq2xs_grid[d & 511], iq2xs_grid[e & 511]);
        auto s = _mm256_setr_epi64x(ksigns64[a >> 9], ksigns64[c >> 9], ksigns64[d >> 9], ksigns64[e >> 9]);
        w = _mm256_sign_epi8(w, _mm256_or_si256(s, _mm256_set1_epi8(1)));
    } else {
        const uint8_t * q = b + 2 + 8 * j;
        const unsigned sc = u32(b + 66 + 4 * j);
        w = _mm256_setr_epi32(iq3xxs_grid[q[0]], iq3xxs_grid[q[1]], iq3xxs_grid[q[2]], iq3xxs_grid[q[3]],
                              iq3xxs_grid[q[4]], iq3xxs_grid[q[5]], iq3xxs_grid[q[6]], iq3xxs_grid[q[7]]);
        auto s = _mm256_setr_epi64x(ksigns64[sc & 127], ksigns64[(sc >> 7) & 127], ksigns64[(sc >> 14) & 127], ksigns64[(sc >> 21) & 127]);
        w = _mm256_sign_epi8(w, _mm256_or_si256(s, _mm256_set1_epi8(1)));
    }
    return _mm256_xor_si256(w, _mm256_set1_epi8(char(128)));
}
static inline __m256i idiv2(__m256i x) { return _mm256_srai_epi32(_mm256_add_epi32(x, _mm256_srli_epi32(x, 31)), 1); } // C x/2
template <int ty, int nb, int ng = 16> static float fast_row2(const uint8_t * r, const RowInput * y) {
    __m256 p[ng];
    for (auto & v : p) v = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const auto * w = r + b * blockbytes(ty);
        __m256i v[8];
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) v[j] = dp(wdec<ty>(w, j), y->q[b * 8 + j]);
        const __m256i h01 = _mm256_hadd_epi32(v[0], v[1]), h23 = _mm256_hadd_epi32(v[2], v[3]);
        const __m256i h45 = _mm256_hadd_epi32(v[4], v[5]), h67 = _mm256_hadd_epi32(v[6], v[7]);
        const __m256i g0 = _mm256_hadd_epi32(h01, h23), g1 = _mm256_hadd_epi32(h45, h67); // [a0..3 | c0..3], [a4..7 | c4..7]
        const __m256i a = _mm256_sub_epi32(_mm256_permute2x128_si256(g0, g1, 0x20), _mm256_load_si256((const __m256i *) &y->corr_lo[b * 8]));
        const __m256i c = _mm256_sub_epi32(_mm256_permute2x128_si256(g0, g1, 0x31), _mm256_load_si256((const __m256i *) &y->corr_hi[b * 8]));
        const __m256i total = _mm256_add_epi32(a, c);
        __m256i sum;
        if constexpr (ty == 18) {
            alignas(32) uint32_t aux[8];
            for (int j = 0; j < 8; ++j) aux[j] = u32(w + 66 + 4 * j) >> 28;
            const __m256i scl = _mm256_load_si256((const __m256i *) aux);
            sum = idiv2(_mm256_add_epi32(_mm256_mullo_epi32(scl, total), idiv2(total)));
        } else {
            const __m256i sc = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) (w + 66)));
            sum = _mm256_add_epi32(idiv2(total), _mm256_add_epi32(_mm256_mullo_epi32(a, _mm256_and_si256(sc, _mm256_set1_epi32(15))),
                                                                  _mm256_mullo_epi32(c, _mm256_srli_epi32(sc, 4))));
            auto bias = _mm256_and_si256(_mm256_srai_epi32(sum, 31), _mm256_set1_epi32(3));
            sum = _mm256_srai_epi32(_mm256_add_epi32(sum, bias), 2);
        }
        auto d = clean(_mm256_mul_ps(_mm256_set1_ps(half(u16(w))), y->scale[b]));
        p[b % ng] = clean(_mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sum), p[b % ng]));
    }
    return _mm_cvtss_f32(_mm256_castps256_ps128(reduce_groups<ng>(p)));
}
static const bool row_v1 = getenv("F53_ROW") != nullptr && getenv("F53_ROW")[0] == '1';
// moe: the one-warp MoE kernel's order (an engine running GGML_CUDA_BATCH_INVARIANT=1); otherwise the four-warp
// order (GGML_CUDA_MMVQ_EXACT_COLS=1 without batch invariance, windows nd0-nd9). Per request (the wire magic);
// F53_ORDER=moe sets it for check and bench.
static const bool env_moe = getenv("F53_ORDER") != nullptr && strcmp(getenv("F53_ORDER"), "moe") == 0;
template <int ty, int nb> static inline float row_sel(const uint8_t * r, const RowInput * y, bool moe) {
    if (moe) return row_v1 ? fast_row<ty, nb, 4>(r, y) : fast_row2<ty, nb, 4>(r, y);
    return row_v1 ? fast_row<ty, nb>(r, y) : fast_row2<ty, nb>(r, y);
}
// fused GLU of the GB10 path: x_bias/gate_bias placeholders (+0), then ggml_cuda_op_swiglu_clamp_single
static inline float swiglu_clamp(float g, float u, const mufu_tables & t) {
    g = add(g, 0.f);
    u = add(u, 0.f);
    g = fminf(g, LIMIT);
    u = fmaxf(fminf(u, LIMIT), -LIMIT);
    return silu(g, u, t);
}

static bool load_mufu(const char * dir, mufu_tables * out) {
    *out = {};
    const char * names[] = { "rcp.bin", "ex2p.bin", "ex2n.bin" };
    size_t sizes[] = { size_t(1) << 25, size_t(31) << 25, size_t(33) << 25 };
    const uint32_t ** dst[] = { &out->rcp, &out->ex2p, &out->ex2n };
    for (int i = 0; i < 3; i++) {
        char path[4096];
        snprintf(path, sizeof path, "%s/%s", dir, names[i]);
        int fd = open(path, O_RDONLY);
        struct stat st {};
        if (fd < 0) return false;
        bool ok = !fstat(fd, &st) && size_t(st.st_size) == sizes[i];
        // anonymous and locked: a file mapping is page cache, which a large read evicts (major faults on the hot path)
        uint8_t * p = ok ? (uint8_t *) aligned_alloc(4096, sizes[i]) : nullptr;
        size_t got = 0;
        while (p && got < sizes[i]) { ssize_t r = read(fd, p + got, sizes[i] - got); if (r <= 0) { ok = false; break; } got += (size_t) r; }
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
        if (!ok || !p) return false;
        if (mlock(p, sizes[i]) != 0) fprintf(stderr, "f53s: mlock of %s failed (%s)\n", names[i], strerror(errno));
        *dst[i] = (const uint32_t *) p;
    }
    return true;
}

// ---- the compute pool: one request at a time, two phases, 32-row tasks; no allocation after create ----
struct Pair { const uint8_t * rec; int tok; float * out; };
struct Pool {
    mufu_tables mt;
    Pair pr[KMAX];
    int k = 0, ntok = 0, phase = 0;
    bool moe = env_moe;   // row order of the request being run (set by the caller before pool_run)
    Q8 qx[TMAX][D / 32];
    Q8 qh[KMAX][F / 32];
    RowInput ix[TMAX], ih[KMAX];
    float h[KMAX][F];
    std::vector<std::thread> threads;
    alignas(64) std::atomic<int> epoch{ 0 };
    alignas(64) std::atomic<int> next{ 0 };
    alignas(64) std::atomic<int> done{ 0 };
    std::atomic<int> ready{ 0 };
    bool stop = false;
};
static void work(Pool * p) {
    const int chunks = p->phase ? D / 32 : F / 32;
    for (int n; (n = p->next.fetch_add(1, std::memory_order_relaxed)) < p->k * chunks;) {
        const int e = n / chunks, start = (n % chunks) * 32;
        const Pair & v = p->pr[e];
        const bool moe = p->moe;
        if (!p->phase) {
            const RowInput * x = &p->ix[v.tok];
            for (int i = start; i < start + 32; i++) {
                const float g = row_sel<17, NBG>(v.rec + size_t(i) * NBG * 74, x, moe);
                const float u = row_sel<17, NBG>(v.rec + GSZ + size_t(i) * NBG * 74, x, moe);
                p->h[e][i] = swiglu_clamp(g, u, p->mt);
            }
            quant(p->h[e] + start, p->qh[e][start / 32], p->mt);
        } else {
            const RowInput * hh = &p->ih[e];
            for (int i = start; i < start + 32; i++) v.out[i] = row_sel<18, NBD>(v.rec + GSZ + USZ + size_t(i) * NBD * 98, hh, moe);
        }
    }
}
__attribute__((noinline)) static void prefault_stack() {
    volatile uint8_t pages[64 * 1024];
    for (size_t i = 0; i < sizeof pages; i += 4096) pages[i] = 0;
}
static Pool * pool_create(const mufu_tables & mt, const std::vector<int> & cpus, int spin_us) {
    auto * p = new Pool;
    memset(p->qx, 0, sizeof p->qx); memset(p->qh, 0, sizeof p->qh); memset(p->h, 0, sizeof p->h);
    memset(p->ix, 0, sizeof p->ix); memset(p->ih, 0, sizeof p->ih);
    p->mt = mt;
    const int n = (int) cpus.size();
    for (int i = 0; i < n; i++) {
        const int cpu = cpus[i];
        p->threads.emplace_back([p, cpu, spin_us] {
            if (cpu >= 0) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }
            prefault_stack();
            { RowInput in{}; uint8_t raw[16 * 98]{}; volatile float r = fast_row<17, NBG>(raw, &in); r = fast_row<18, NBD>(raw, &in);
              r = fast_row2<17, NBG>(raw, &in); r = fast_row2<18, NBD>(raw, &in);
              r = fast_row<17, NBG, 4>(raw, &in); r = fast_row<18, NBD, 4>(raw, &in);
              r = fast_row2<17, NBG, 4>(raw, &in); r = fast_row2<18, NBD, 4>(raw, &in); (void) r; }
            errno = 0;
            int last = 0;
            auto idle = std::chrono::steady_clock::now();
            p->ready.fetch_add(1, std::memory_order_release);
            for (;;) {
                const int ep = p->epoch.load(std::memory_order_acquire);
                if (ep == last) {
                    if (spin_us < 0 || std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - idle).count() < spin_us) {
                        _mm_pause();
                        continue;
                    }
                    syscall(SYS_futex, reinterpret_cast<int *>(&p->epoch), FUTEX_WAIT_PRIVATE, last, nullptr, nullptr, 0);
                    continue;
                }
                last = ep;
                if (p->stop) return;
                work(p);
                p->done.fetch_add(1, std::memory_order_release);
                idle = std::chrono::steady_clock::now();
            }
        });
    }
    while (p->ready.load(std::memory_order_acquire) != n) _mm_pause();
    return p;
}
// x: ntok inputs of D floats; pairs: k (rec, tok, out)
static void pool_run(Pool * p, const float * x, int ntok, const Pair * pairs, int k) {
    p->k = k;
    p->ntok = ntok;
    for (int i = 0; i < k; i++) p->pr[i] = pairs[i];
    for (int t = 0; t < ntok; t++) {
        for (int b = 0; b < D / 32; b++) quant(x + size_t(t) * D + 32 * b, p->qx[t][b], p->mt);
        prepare(p->ix[t], p->qx[t], NBG);
    }
    for (int phase = 0; phase < 2; phase++) {
        if (phase) for (int i = 0; i < k; ++i) prepare(p->ih[i], p->qh[i], NBD);
        p->phase = phase;
        p->next.store(0, std::memory_order_relaxed);
        p->done.store(0, std::memory_order_relaxed);
        p->epoch.fetch_add(1, std::memory_order_release);
        syscall(SYS_futex, reinterpret_cast<int *>(&p->epoch), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
        work(p); // the caller works too
        while (p->done.load(std::memory_order_acquire) != (int) p->threads.size()) _mm_pause();
    }
}

static std::vector<int> parse_cpus(const char * s) {
    std::vector<int> v;
    if (!s || !*s) { for (int i = 0; i < 4; i++) v.push_back(-1); return v; }
    while (*s) {
        char * e; long a = strtol(s, &e, 10), b = a;
        if (*e == '-') b = strtol(e + 1, &e, 10);
        for (long i = a; i <= b; i++) v.push_back((int) i);
        s = *e == ',' ? e + 1 : e;
    }
    return v;
}
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static uint8_t * read_file(const char * path, size_t & n) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st {};
    fstat(fd, &st);
    n = (size_t) st.st_size;
    uint8_t * p = (uint8_t *) aligned_alloc(4096, (n + 4095) & ~size_t(4095));
    size_t got = 0;
    while (got < n) {
        const size_t want = std::min(n - got, (size_t) 256 << 20);
        ssize_t r = read(fd, p + got, want);
        if (r <= 0) { perror("read"); exit(1); }
        posix_fadvise(fd, (off_t) got, r, POSIX_FADV_DONTNEED); // keep the page cache from pushing anything else out
        got += (size_t) r;
    }
    close(fd);
    return p;
}

static int cmd_check(Pool * pool, const char * recs, const char * frb) {
    size_t nr, nf;
    uint8_t * R = read_file(recs, nr);
    uint8_t * B = read_file(frb, nf);
    if (nr != 8 * REC) { fprintf(stderr, "recs8 size %zu != %zu\n", nr, 8 * REC); return 1; }
    const int * hdr = (const int *) B;
    const int nc = hdr[0], T = hdr[1], NE = hdr[2];
    const uint8_t * c = B + 12;
    std::vector<float> out((size_t) KMAX * D);
    long rows = 0, bad = 0, badv = 0;
    double tsum = 0;
    for (int ci = 0; ci < nc; ci++) {
        const float * x = (const float *) c; c += size_t(T) * D * 4;
        const int * ids = (const int *) c; c += size_t(T) * NE * 4;
        const float * ref = (const float *) c; c += size_t(T) * NE * D * 4;
        Pair pr[KMAX];
        int k = 0;
        for (int t = 0; t < T; t++) for (int j = 0; j < NE; j++) { pr[k] = { R + size_t(ids[t * NE + j]) * REC, t, out.data() + size_t(k) * D }; k++; }
        // k = T*8 may exceed KMAX: run in chunks of KMAX
        double t0 = now_us();
        for (int o = 0; o < k; o += KMAX) pool_run(pool, x, T, pr + o, std::min(KMAX, k - o));
        tsum += now_us() - t0;
        for (int i = 0; i < k; i++) {
            const float * a = pr[i].out, * b = ref + size_t(i) * D;
            int nb = 0;
            for (int j = 0; j < D; j++) nb += bits(a[j]) != bits(b[j]);
            rows++;
            if (nb) {
                if (bad < 5) {
                    int j = 0; while (bits(a[j]) == bits(b[j])) j++;
                    fprintf(stderr, "case %d pair %d (tok %d exp %d): %d values differ, first [%d] %.9g vs ref %.9g\n", ci, i, pr[i].tok, ids[i], nb, j, a[j], b[j]);
                }
                bad++; badv += nb;
            }
        }
    }
    printf("check %s: %d cases T=%d, %ld rows, %ld rows differ (%ld values), %.1f us per case\n", frb, nc, T, rows, bad, badv, tsum / nc);
    return bad ? 2 : 0;
}

static int cmd_bench(Pool * pool, const char * recs, int k, int ntok, int reps) {
    size_t nr;
    uint8_t * R = read_file(recs, nr);
    std::vector<float> x((size_t) TMAX * D), out((size_t) KMAX * D);
    for (size_t i = 0; i < x.size(); i++) x[i] = float((i * 2654435761u) % 2001) / 1000.0f - 1.0f;
    Pair pr[KMAX];
    for (int i = 0; i < k; i++) pr[i] = { R + size_t(i % 8) * REC, i % ntok, out.data() + size_t(i) * D };
    std::vector<double> t;
    for (int r = 0; r < reps; r++) { double t0 = now_us(); pool_run(pool, x.data(), ntok, pr, k); t.push_back(now_us() - t0); }
    std::sort(t.begin(), t.end());
    printf("bench k=%d ntok=%d: p50 %.0f us p90 %.0f us min %.0f us (%.2f GB/s of records)\n", k, ntok, t[t.size() / 2], t[t.size() * 9 / 10], t[0],
           k * (double) REC / t[t.size() / 2] / 1e3);
    return 0;
}

static bool recv_all(int fd, void * buf, size_t n, int spin_us) {
    uint8_t * p = (uint8_t *) buf;
    size_t got = 0;
    double t_idle = now_us();
    while (got < n) {
        ssize_t r = recv(fd, p + got, n - got, MSG_DONTWAIT);
        if (r > 0) { got += (size_t) r; t_idle = now_us(); continue; }
        if (r == 0) return false;
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return false;
        if (now_us() - t_idle > spin_us) { pollfd pf = { fd, POLLIN, 0 }; poll(&pf, 1, 100); }
        else _mm_pause();
    }
    return true;
}
static bool send_all(int fd, const void * buf, size_t n) {
    const uint8_t * p = (const uint8_t *) buf;
    while (n) { ssize_t r = send(fd, p, n, MSG_NOSIGNAL); if (r <= 0) { if (errno == EINTR) continue; return false; } p += r; n -= (size_t) r; }
    return true;
}

static int cmd_serve(Pool * pool, const char * recs, const char * idx, int port, int spin_us) {
    size_t nr;
    uint8_t * R = read_file(recs, nr);
    {   // lock as much of the records as RLIMIT_MEMLOCK allows (after the tables); the rest stays anonymous memory
        rlimit rl{}; getrlimit(RLIMIT_MEMLOCK, &rl);
        size_t lim = rl.rlim_cur == RLIM_INFINITY ? nr : (size_t) rl.rlim_cur;
        const size_t tables = (size_t(65) << 25) + (64u << 20);
        size_t want = lim > tables ? std::min(nr, (lim - tables) & ~size_t(4095)) : 0;
        const bool ok = want > 0 && mlock(R, want) == 0;
        fprintf(stderr, "f53s: locked %.1f of %.1f GB of records (memlock limit %.1f GB)\n", ok ? want / 1e9 : 0.0, nr / 1e9, lim / 1e9);
    }
    std::vector<const uint8_t *> by_gid(size_t(NLAYER) * NEXP, nullptr);
    FILE * f = fopen(idx, "r");
    if (!f) { perror(idx); return 1; }
    long gid; long long off; size_t n = 0;
    while (fscanf(f, "%ld %lld", &gid, &off) == 2) {
        if (gid < 0 || gid >= NLAYER * NEXP || off < 0 || size_t(off) + REC > nr) { fprintf(stderr, "bad idx line %ld %lld\n", gid, off); return 1; }
        by_gid[(size_t) gid] = R + off; n++;
    }
    fclose(f);
    fprintf(stderr, "f53s: %zu records (%.1f GB), %zu workers + caller, port %d\n", n, nr / 1e9, pool->threads.size(), port);
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t) port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(ls, (sockaddr *) &a, sizeof a) || listen(ls, 4)) { perror("bind/listen"); return 1; }
    int32_t hdr[4 + 2 * KMAX];
    std::vector<float> x((size_t) TMAX * D), out((size_t) KMAX * D);
    const std::vector<float> nanrow(D, __builtin_nanf(""));
    uint64_t nreq = 0, npairs = 0, nnan = 0, nmoe = 0; double tc = 0;
    std::vector<float> tlat(1 << 16, 0.f); // ring of per-request compute times (us)
    rusage ru0{}; getrusage(RUSAGE_SELF, &ru0);
    for (;;) {
        int fd = accept(ls, nullptr, nullptr);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        fprintf(stderr, "f53s: client connected\n");
        for (;;) {
            if (!recv_all(fd, hdr, sizeof hdr, spin_us)) break;
            const int k = hdr[0] & 0xffff, ntok = hdr[1];
            const uint32_t magic = uint32_t(hdr[0]) & 0xffff0000u;
            if ((magic != MAGIC && magic != MAGIC_MOE) || k < 1 || k > KMAX || ntok < 1 || ntok > TMAX) { fprintf(stderr, "f53s: bad header %08x %d\n", hdr[0], ntok); break; }
            if (!recv_all(fd, x.data(), size_t(ntok) * D * 4, spin_us)) break;
            const double t0 = now_us();
            Pair pr[KMAX]; int kk = 0; int map[KMAX];
            for (int i = 0; i < k; i++) {
                const int g = hdr[4 + 2 * i], t = hdr[5 + 2 * i];
                const uint8_t * r = g >= 0 && g < NLAYER * NEXP && t >= 0 && t < ntok ? by_gid[(size_t) g] : nullptr;
                if (r) { pr[kk] = { r, t, out.data() + size_t(i) * D }; map[i] = kk++; } else { map[i] = -1; nnan++; }
            }
            pool->moe = magic == MAGIC_MOE;
            nmoe += magic == MAGIC_MOE;
            if (kk) pool_run(pool, x.data(), ntok, pr, kk);
            const double dt = now_us() - t0;
            tc += dt;
            tlat[nreq % tlat.size()] = (float) (dt / std::max(kk, 1));
            for (int i = 0; i < k; i++) if (map[i] < 0) memcpy(out.data() + size_t(i) * D, nanrow.data(), D * 4);
            if (!send_all(fd, out.data(), size_t(k) * D * 4)) break;
            nreq++; npairs += (uint64_t) k;
            if (nreq % 20000 == 0) {
                rusage ru{}; getrusage(RUSAGE_SELF, &ru);
                fprintf(stderr, "f53s: %llu requests, %.2f pairs per request, compute %.0f us per request, %llu NaN rows, minor faults %ld major %ld since start\n",
                        (unsigned long long) nreq, (double) npairs / nreq, tc / nreq, (unsigned long long) nnan, ru.ru_minflt - ru0.ru_minflt, ru.ru_majflt - ru0.ru_majflt);
            }
        }
        close(fd);
        {
            std::vector<float> v(tlat.begin(), tlat.begin() + std::min<size_t>(nreq, tlat.size()));
            std::sort(v.begin(), v.end());
            if (!v.empty()) fprintf(stderr, "f53s: compute per pair p50 %.0f us p90 %.0f us p99 %.0f us (last %zu requests)\n", v[v.size() / 2], v[v.size() * 9 / 10], v[v.size() * 99 / 100], v.size());
        }
        rusage ru{}; getrusage(RUSAGE_SELF, &ru);
        fprintf(stderr, "f53s: client gone after %llu requests, %llu in the one-warp MoE order (%.2f pairs per request, compute %.0f us per request, %llu NaN rows; minor faults %ld major %ld since start)\n",
                (unsigned long long) nreq, (unsigned long long) nmoe, nreq ? (double) npairs / nreq : 0.0, nreq ? tc / nreq : 0.0, (unsigned long long) nnan, ru.ru_minflt - ru0.ru_minflt, ru.ru_majflt - ru0.ru_majflt);
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: f53s serve recs idx port | check recs8 frb | bench recs8 k ntok reps\n"); return 1; }
    const char * md = getenv("F53_MUFU");
    mufu_tables mt;
    if (!md || !load_mufu(md, &mt)) { fprintf(stderr, "F53_MUFU must name a directory with the verified GB10 rcp/ex2p/ex2n tables\n"); return 1; }
    const int spin_us = getenv("F53_SPIN_US") ? atoi(getenv("F53_SPIN_US")) : 2000;
    std::vector<int> cpus = parse_cpus(getenv("F53_CPUS"));
    // the caller is one of the workers: pin it to the first cpu, the pool gets the rest
    if (!cpus.empty() && cpus[0] >= 0) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus[0], &s); sched_setaffinity(0, sizeof s, &s); }
    std::vector<int> wc(cpus.begin() + (cpus.size() > 1 ? 1 : 0), cpus.end());
    if (cpus.size() <= 1) wc.clear();
    Pool * pool = pool_create(mt, wc, spin_us);
    const std::string cmd = argv[1];
    if (cmd == "check" && argc >= 4) return cmd_check(pool, argv[2], argv[3]);
    if (cmd == "bench" && argc >= 6) return cmd_bench(pool, argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    if (cmd == "serve" && argc >= 5) return cmd_serve(pool, argv[2], argv[3], atoi(argv[4]), spin_us);
    fprintf(stderr, "bad command\n");
    return 1;
}
