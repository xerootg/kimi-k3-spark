#include "g53cpu.h"
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_SYCL
#include "../ref/ggml-common.h"
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <fcntl.h>
#include <immintrin.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>
using namespace g53exact;
static constexpr int D = 6144, F = 2048;
static int sum4(__m128i v) {
  v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
  v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
  return _mm_cvtsi128_si32(v);
}
static __m256i dp(__m256i a, __m256i b) {
#ifdef __AVXVNNI__
  return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), a, b);
#else
  // Widen before multiplying: maddubs would saturate for arbitrary signed q8.
  __m256i lo =
      _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(a)),
                         _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b)));
  __m256i hi =
      _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(a, 1)),
                         _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1)));
  __m256i one = _mm256_set1_epi16(1);
  lo = _mm256_madd_epi16(lo, one);
  hi = _mm256_madd_epi16(hi, one);
  return _mm256_set_m128i(_mm_hadd_epi32(_mm256_castsi256_si128(hi),
                                         _mm256_extracti128_si256(hi, 1)),
                          _mm_hadd_epi32(_mm256_castsi256_si128(lo),
                                         _mm256_extracti128_si256(lo, 1)));
#endif
}
template <int ty> static inline int isum(const uint8_t *b, int j, const Q8 &y) {
  __m256i w;
  unsigned sc = 0;
  if (ty == 17) {
    const uint8_t *q = b + 2 + 8 * j;
    unsigned a = u16(q), c = u16(q + 2), d = u16(q + 4), e = u16(q + 6);
    w = _mm256_setr_epi64x(iq2xs_grid[a & 511], iq2xs_grid[c & 511],
                           iq2xs_grid[d & 511], iq2xs_grid[e & 511]);
    auto s = _mm256_setr_epi64x(ksigns64[a >> 9], ksigns64[c >> 9],
                                ksigns64[d >> 9], ksigns64[e >> 9]);
    w = _mm256_sign_epi8(w, _mm256_or_si256(s, _mm256_set1_epi8(1)));
    sc = b[66 + j];
  } else if (ty == 18) {
    const uint8_t *q = b + 2 + 8 * j;
    sc = u32(b + 66 + 4 * j);
    w = _mm256_setr_epi32(iq3xxs_grid[q[0]], iq3xxs_grid[q[1]],
                          iq3xxs_grid[q[2]], iq3xxs_grid[q[3]],
                          iq3xxs_grid[q[4]], iq3xxs_grid[q[5]],
                          iq3xxs_grid[q[6]], iq3xxs_grid[q[7]]);
    auto s = _mm256_setr_epi64x(ksigns64[sc & 127], ksigns64[(sc >> 7) & 127],
                                ksigns64[(sc >> 14) & 127],
                                ksigns64[(sc >> 21) & 127]);
    w = _mm256_sign_epi8(w, _mm256_or_si256(s, _mm256_set1_epi8(1)));
  } else {
    auto q = _mm_loadu_si128((const __m128i *)(b + 8 + 16 * j));
    auto idx = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q),
                                _mm256_set1_epi8(15));
    auto table = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    w = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(table), idx);
    sc = ((b[4 + j / 2] >> (4 * (j & 1))) & 15) |
         (((u16(b + 2) >> (2 * j)) & 3) << 4);
  }
  // Offset signed weights into unsigned bytes, and subtract 128*sum(q).
  // Keeping signs on weights also handles q=-128 without byte negation
  // overflow.
  auto yy = _mm256_loadu_si256((const __m256i *)y.q);
  auto v = dp(_mm256_xor_si256(w, _mm256_set1_epi8(char(128))), yy);
  v = _mm256_sub_epi32(v, _mm256_slli_epi32(dp(_mm256_set1_epi8(1), yy), 7));
  int a = sum4(_mm256_castsi256_si128(v)),
      c = sum4(_mm256_extracti128_si256(v, 1));
  return ty == 17   ? (a * int(sc & 15) + c * int(sc >> 4) + (a + c) / 2) / 4
         : ty == 18 ? (int(sc >> 28) * (a + c) + (a + c) / 2) / 2
                    : (a + c) * (int(sc) - 32);
}
static __m256 clean(__m256 x) {
  auto u = _mm256_castps_si256(x),
       a = _mm256_and_si256(u, _mm256_set1_epi32(0x7fffffff));
  auto sub = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800000), a);
  u = _mm256_blendv_epi8(
      u, _mm256_and_si256(u, _mm256_set1_epi32(int(0x80000000))), sub);
  return _mm256_castsi256_ps(
      _mm256_blendv_epi8(u, _mm256_set1_epi32(0x7fffffff),
                         _mm256_cmpgt_epi32(a, _mm256_set1_epi32(0x7f800000))));
}
// p[b] packs virtual threads 8*b..8*b+7. b+16 revisits the same
// accumulator, retaining each CUDA thread's increasing-kb FMA sequence.
template <int ty, int nb> static float fast_row(const uint8_t *r, const Q8 *y) {
  __m256 p[16];
  for (auto &v : p)
    v = _mm256_setzero_ps();
  for (int b = 0; b < nb; b++) {
    alignas(32) uint16_t hd[8];
    alignas(32) int s[8];
    const auto *w = r + b * blockbytes(ty);
    for (int j = 0; j < 8; j++) {
      hd[j] = y[b * 8 + j].d;
      s[j] = isum<ty>(w, j, y[b * 8 + j]);
    }
    auto d = clean(
        _mm256_mul_ps(_mm256_set1_ps(half(u16(w))),
                      _mm256_cvtph_ps(_mm_load_si128((const __m128i *)hd))));
    p[b % 16] = clean(_mm256_fmadd_ps(
        d, _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i *)s)),
        p[b % 16]));
  }
  // Each group of four b vectors is a CUDA warp. Add w0,w1,w2,w3
  // in order, then apply XOR offsets 16,8,4,2,1 (lane-zero subtree).
  __m256 lane[4];
  for (int i = 0; i < 4; i++)
    lane[i] = clean(_mm256_add_ps(
        clean(_mm256_add_ps(clean(_mm256_add_ps(p[i], p[i + 4])), p[i + 8])),
        p[i + 12]));
  lane[0] = clean(_mm256_add_ps(lane[0], lane[2]));
  lane[1] = clean(_mm256_add_ps(lane[1], lane[3]));
  lane[0] = clean(_mm256_add_ps(lane[0], lane[1]));
  alignas(32) float s[8];
  _mm256_store_ps(s, lane[0]);
  for (int off = 4; off; off >>= 1)
    for (int i = 0; i < off; i++)
      s[i] = add(s[i], s[i + off]);
  return s[0];
}
float g53cpu_row(int ty, const uint8_t *r, int nb, const Q8 *y) {
#define DISPATCH(T)                                                            \
  case T:                                                                      \
    return nb == 8 ? fast_row<T, 8>(r, y) : fast_row<T, 24>(r, y)
  if (nb != 8 && nb != 24)
    abort();
  switch (ty) {
    DISPATCH(17);
    DISPATCH(18);
    DISPATCH(23);
  default:
    abort();
  }
#undef DISPATCH
}
size_t g53cpu_rec_bytes(int a, int b, int c, int layout) {
  if (layout || !blockbytes(a) || !blockbytes(b) || !blockbytes(c))
    return 0;
  return size_t(D / 256) * F * (blockbytes(a) + blockbytes(b) + blockbytes(c));
}
bool g53cpu_load_mufu(const char *dir, g53_mufu_tabs *out) {
  *out = {};
  const char *names[] = {"rcp.bin", "ex2p.bin", "ex2n.bin"};
  size_t sizes[] = {size_t(1) << 25, size_t(31) << 25, size_t(33) << 25};
  const uint32_t **dst[] = {&out->rcp, &out->ex2p, &out->ex2n};
  for (int i = 0; i < 3; i++) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, names[i]);
    int fd = open(path, O_RDONLY);
    struct stat st{};
    if (fd < 0)
      return false;
    bool ok = !fstat(fd, &st) && size_t(st.st_size) == sizes[i];
    void *p = ok ? mmap(nullptr, sizes[i], PROT_READ,
                        MAP_PRIVATE | MAP_POPULATE, fd, 0)
                 : MAP_FAILED;
    close(fd);
    if (p == MAP_FAILED)
      return false;
    *dst[i] = (const uint32_t *)p;
  }
  return true;
}
struct g53cpu_pool {
  g53_mufu_tabs mt;
  g53cpu_expert e[16];
  Q8 qx[D / 32], qh[16][F / 32];
  float h[16][F];
  std::vector<std::thread> threads;
  std::atomic<int> epoch{0}, next{0}, done{0};
  bool stop = false;
  int phase = 0, k = 0;
  bool scalar = false;
};
static float rowfn(g53cpu_pool *p, int t, const uint8_t *r, int nb,
                   const Q8 *y) {
  return p->scalar
             ? row(t, r, nb, y, false, iq2xs_grid, iq3xxs_grid, kvalues_iq4nl)
             : g53cpu_row(t, r, nb, y);
}
static void work(g53cpu_pool *p) {
  int chunks = p->phase ? D / 32 : F / 32;
  for (int n; (n = p->next.fetch_add(1)) < p->k * chunks;) {
    int e = n / chunks, start = (n % chunks) * 32;
    auto &v = p->e[e];
    size_t gs = size_t(F) * 24 * blockbytes(v.tg),
           us = size_t(F) * 24 * blockbytes(v.tu);
    if (!p->phase) {
      for (int i = start; i < start + 32; i++) {
        float g = rowfn(p, v.tg, v.rec + i * 24 * blockbytes(v.tg), 24, p->qx),
              u = rowfn(p, v.tu, v.rec + gs + i * 24 * blockbytes(v.tu), 24,
                        p->qx);
        if (v.g)
          v.g[i] = g;
        if (v.u)
          v.u[i] = u;
        p->h[e][i] = silu(g, u, p->mt);
        if (v.h)
          v.h[i] = p->h[e][i];
      }
      quant(p->h[e] + start, p->qh[e][start / 32], p->mt);
    } else
      for (int i = start; i < start + 32; i++)
        v.out[i] = rowfn(p, v.td, v.rec + gs + us + i * 8 * blockbytes(v.td), 8,
                         p->qh[e]);
  }
}
g53cpu_pool *g53cpu_create(const g53_mufu_tabs *mt, const g53cpu_opts &o) {
  if (o.ncpu < 1 || o.ncpu > 256)
    return nullptr;
  for (int i = 0; o.cpus && i < o.ncpu; i++)
    if (o.cpus[i] < 0 || o.cpus[i] >= CPU_SETSIZE)
      return nullptr;
  auto *p = new g53cpu_pool;
  memset(p->qx, 0, sizeof p->qx);
  memset(p->qh, 0, sizeof p->qh);
  memset(p->h, 0, sizeof p->h);
  p->mt = *mt;
  p->scalar = getenv("G53CPU_SCALAR") != nullptr;
  for (int i = 0; i < o.ncpu; i++) {
    const int cpu = o.cpus ? o.cpus[i] : -1;
    p->threads.emplace_back([p, i, o, cpu] {
      if (o.rt_prio > 0) {
        sched_param sp{};
        sp.sched_priority = o.rt_prio;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp))
          fprintf(stderr, "g53cpu: worker %d SCHED_FIFO unavailable\n", i);
      }
      if (cpu >= 0) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(cpu, &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
      }
      int last = 0;
      auto idle = std::chrono::steady_clock::now();
      for (;;) {
        int ep = p->epoch.load(std::memory_order_acquire);
        if (ep == last) {
          if (o.spin_us < 0 ||
              (o.spin_us > 0 &&
               std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - idle)
                       .count() < o.spin_us)) {
            _mm_pause();
            continue;
          }
          syscall(SYS_futex, reinterpret_cast<int *>(&p->epoch),
                  FUTEX_WAIT_PRIVATE, last, nullptr, nullptr, 0);
          continue;
        }
        last = ep;
        if (p->stop)
          return;
        work(p);
        p->done.fetch_add(1, std::memory_order_release);
        idle = std::chrono::steady_clock::now();
      }
    });
  }
  return p;
}
void g53cpu_destroy(g53cpu_pool *p) {
  p->stop = true;
  p->epoch.fetch_add(1, std::memory_order_release);
  syscall(SYS_futex, reinterpret_cast<int *>(&p->epoch), FUTEX_WAKE_PRIVATE,
          INT_MAX, nullptr, nullptr, 0);
  for (auto &t : p->threads)
    t.join();
  delete p;
}
void g53cpu_submit(g53cpu_pool *p, const float *x, const g53cpu_expert *e,
                   int k) {
  if (k < 1 || k > 16)
    abort();
  p->k = k;
  for (int i = 0; i < k; i++) {
    if (!g53cpu_rec_bytes(e[i].tg, e[i].tu, e[i].td, e[i].layout))
      abort();
    p->e[i] = e[i];
  }
  for (int b = 0; b < D / 32; b++)
    quant(x + 32 * b, p->qx[b], p->mt);
  for (int phase = 0; phase < 2; phase++) {
    p->phase = phase;
    p->next = 0;
    p->done = 0;
    p->epoch.fetch_add(1, std::memory_order_release);
    syscall(SYS_futex, reinterpret_cast<int *>(&p->epoch), FUTEX_WAKE_PRIVATE,
            INT_MAX, nullptr, nullptr, 0);
    while (p->done.load(std::memory_order_acquire) != int(p->threads.size()))
      _mm_pause();
  }
}
bool g53cpu_done(g53cpu_pool *, int) { return true; }
void g53cpu_wait(g53cpu_pool *, int) {}
void g53cpu_wait_all(g53cpu_pool *) {}
void g53cpu_run(g53cpu_pool *p, const float *x, const g53cpu_expert *e, int k) {
  g53cpu_submit(p, x, e, k);
}
void g53cpu_last_times(g53cpu_pool *, g53cpu_times *t) { *t = {}; }
void g53cpu_reference(const uint8_t *r, int a, int b, int c, const float *x,
                      const g53_mufu_tabs &t, float *out) {
  Q8 qx[D / 32], qh[F / 32];
  float h[F];
  for (int i = 0; i < D / 32; i++)
    quant(x + 32 * i, qx[i], t);
  size_t gs = size_t(F) * 24 * blockbytes(a),
         us = size_t(F) * 24 * blockbytes(b);
  for (int i = 0; i < F; i++)
    h[i] = silu(row(a, r + i * 24 * blockbytes(a), 24, qx, false, iq2xs_grid,
                    iq3xxs_grid, kvalues_iq4nl),
                row(b, r + gs + i * 24 * blockbytes(b), 24, qx, false,
                    iq2xs_grid, iq3xxs_grid, kvalues_iq4nl),
                t);
  for (int i = 0; i < F / 32; i++)
    quant(h + 32 * i, qh[i], t);
  for (int i = 0; i < D; i++)
    out[i] = row(c, r + gs + us + i * 8 * blockbytes(c), 8, qh, false,
                 iq2xs_grid, iq3xxs_grid, kvalues_iq4nl);
}

void g53cpu_set_scalar(g53cpu_pool *p, bool s) { p->scalar = s; }
