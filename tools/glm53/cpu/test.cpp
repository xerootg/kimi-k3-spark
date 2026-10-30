#define main g53_scalar_driver_main
#include "../g53ref_exact.cpp"
#undef main
#include "g53cpu.h"
#include "mapped_fixture.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <sys/prctl.h>
#include <sys/resource.h>
static std::atomic<bool> counting{false};
static std::atomic<long> allocs{0};
extern "C" void *__real_malloc(size_t);
extern "C" void *__real_calloc(size_t, size_t);
extern "C" void *__real_realloc(void *, size_t);
extern "C" int __real_posix_memalign(void **, size_t, size_t);
extern "C" void *__real_aligned_alloc(size_t, size_t);
extern "C" void *__wrap_malloc(size_t n) {
  if (counting)
    allocs++;
  return __real_malloc(n);
}
extern "C" void *__wrap_calloc(size_t a, size_t b) {
  if (counting)
    allocs++;
  return __real_calloc(a, b);
}
extern "C" void *__wrap_realloc(void *p, size_t n) {
  if (counting)
    allocs++;
  return __real_realloc(p, n);
}
extern "C" int __wrap_posix_memalign(void **p, size_t a, size_t n) {
  if (counting)
    allocs++;
  return __real_posix_memalign(p, a, n);
}
extern "C" void *__wrap_aligned_alloc(size_t a, size_t n) {
  if (counting)
    allocs++;
  return __real_aligned_alloc(a, n);
}
void *operator new(size_t n) {
  if (counting)
    allocs++;
  void *p = __real_malloc(n);
  if (!p)
    throw std::bad_alloc();
  return p;
}
void operator delete(void *p) noexcept { free(p); }
void operator delete(void *p, size_t) noexcept { free(p); }
static void equal(const float *a, const float *b, size_t n, const char *name) {
  if (memcmp(a, b, n * 4)) {
    for (size_t i = 0; i < n; i++)
      if (bits(a[i]) != bits(b[i])) {
        std::cerr << name << " mismatch " << i << " " << std::hex << bits(a[i])
                  << " != " << bits(b[i]) << std::dec << "\n";
        break;
      }
    throw std::runtime_error(name);
  }
}
// Exercise prepared pool inputs without external weights or MUFU files.
// Zero gate + integer x with max=127 keeps every MUFU lookup closed-form;
// the nonzero up projection still checks the cached bytes/scales/corrections.
static void closed_pool() {
  std::mt19937 rng(531);
  mufu_tables mt{};
  for (int tu : {17, 18, 23}) {
    const int tg = 17, td = tu;
    size_t rs = g53cpu_rec_bytes(tg, tu, td);
    size_t gs = size_t(F) * 24 * blockbytes(tg);
    std::vector<uint8_t> raw(rs * 8);
    for (auto &v : raw)
      v = rng();
    for (int e = 0; e < 8; ++e) {
      size_t off = rs * e;
      for (int ty : {tg, tu, td})
        for (int b = 0; b < 49152; ++b) {
          raw[off] = 0;
          raw[off + 1] = off < rs * e + gs ? 0 : 0x30;
          off += blockbytes(ty);
        }
    }
    MappedFixture rec(raw);
    std::vector<float> x(D), out(8 * D), up(8 * F), expected(8 * F);
    for (int i = 0; i < D; ++i)
      x[i] = i % 32 == 0 ? 127.f : float(i % 255 - 127);
    Q8 q[D / 32];
    for (int i = 0; i < D / 32; ++i)
      quant(x.data() + 32 * i, q[i], mt);
    g53cpu_expert es[8];
    for (int e = 0; e < 8; ++e) {
      es[e] = {rec.data() + e * rs, tg, tu, td, out.data() + e * D, nullptr,
               up.data() + e * F};
      for (int i = 0; i < F; ++i)
        expected[e * F + i] =
            row(tu, rec.data() + e * rs + gs + i * 24 * blockbytes(tu), 24, q,
                false, iq2xs_grid, iq3xxs_grid, kvalues_iq4nl);
    }
    for (int nt : {4, 8, 12}) {
      g53cpu_opts opts;
      opts.ncpu = nt;
      auto *p = g53cpu_create(&mt, opts);
      for (int k : {1, 4, 8}) {
        for (int warm = 0; warm < 3; ++warm)
          g53cpu_run(p, x.data(), es, k);
        equal(up.data(), expected.data(), k * F, "closed pool up");
        for (int i = 0; i < k * D; ++i)
          require(bits(out[i]) == 0, "closed pool down");
        rusage before{}, after{};
        getrusage(RUSAGE_SELF, &before);
        allocs = 0;
        counting = true;
        for (int repeat = 0; repeat < 3; ++repeat)
          g53cpu_run(p, x.data(), es, k);
        counting = false;
        getrusage(RUSAGE_SELF, &after);
        require(allocs == 0, "closed pool allocations");
        if (after.ru_minflt != before.ru_minflt ||
            after.ru_majflt != before.ru_majflt)
          std::cerr << "fault gate tu=" << tu << " nt=" << nt << " k=" << k
                    << " minor=" << after.ru_minflt - before.ru_minflt
                    << " major=" << after.ru_majflt - before.ru_majflt << "\n";
        require(after.ru_minflt == before.ru_minflt &&
                    after.ru_majflt == before.ru_majflt,
                "closed pool page faults");
      }
      g53cpu_destroy(p);
    }
  }
  std::cout << "PASS closed pool IQ2/3/4 up words; k=1/4/8 threads=4/8/12 "
               "allocations=0 faults=0/0\n";
}
int main(int argc, char **argv) try {
  // Anonymous randomized fixtures must not be collapsed/migrated by khugepaged
  // during a measured request. Production uses prefaulted, locked file
  // mappings.
  require(prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) == 0, "disable fixture THP");
  std::cout.setf(std::ios::unitbuf);
  unit();
  std::mt19937 rng(178);
  for (int ty : {17, 18, 23})
    for (int nb : {8, 24})
      for (int c = 0; c < 100; c++) {
        std::vector<uint8_t> raw(nb * blockbytes(ty));
        std::vector<Q8> q(nb * 8);
        for (auto &v : raw)
          v = rng();
        for (int b = 0; b < nb; b++) {
          uint16_t d = c < 20 ? uint16_t(rng()) : tohalf(.125f);
          memcpy(raw.data() + b * blockbytes(ty), &d, 2);
        }
        for (auto &v : q) {
          for (auto &x : v.q)
            x = int8_t(rng());
          v.d = c < 20 ? uint16_t(rng()) : tohalf(.0625f);
        }
        float a = row(ty, raw.data(), nb, q.data(), false, iq2xs_grid,
                      iq3xxs_grid, kvalues_iq4nl),
              b = g53cpu_row(ty, raw.data(), nb, q.data());
        equal(&a, &b, 1, "synthetic row");
      }
  std::cout << "PASS 600 AVX rows vs g53ref_exact (all q8 bytes, exceptional "
               "half scales, CUDA reduction)\n";
  closed_pool();
  if (argc < 3) {
    std::cout << "SKIP extracted slices and table-dependent end-to-end: pass "
                 "DATA MUFU\n";
    std::cout << "PASS ALL AVAILABLE GLM CPU exact tests\n";
    return 0;
  }
  bool full = std::filesystem::exists(std::string(argv[2]) + "/ex2n.bin");
  Tables mt(full ? argv[2] : "-");
  if (!full)
    std::cout
        << "SKIP table-dependent random SwiGLU: verified MUFU files absent\n";
  for (int layer : {3, 8, 75}) {
    int tg = layer == 8 ? 18 : 17, td = layer == 3 ? 18 : 23;
    Weights w(false,
              std::string(argv[1]) + "/UD-Q2_K_XL_L" + std::to_string(layer), 8,
              tg, tg, td);
    std::vector<uint8_t> rec(g53cpu_rec_bytes(tg, tg, td) * 8);
    size_t rs = rec.size() / 8;
    for (int e = 0; e < 8; e++) {
      memcpy(rec.data() + rs * e, w.ptr(e, 0), w.gs);
      memcpy(rec.data() + rs * e + w.gs, w.ptr(e, 1), w.us);
      memcpy(rec.data() + rs * e + w.gs + w.us, w.ptr(e, 2), w.ds);
    }
    std::vector<float> x(D), out(8 * D), ref(8 * D), gg(8 * F), uu(8 * F),
        hh(8 * F);
    int32_t ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    auto expected = outputs();
    g53cpu_expert es[8];
    for (int e = 0; e < 8; e++)
      es[e] = {rec.data() + rs * e,
               tg,
               tg,
               td,
               out.data() + e * D,
               gg.data() + e * F,
               uu.data() + e * F,
               hh.data() + e * F};
    for (int nt : {4, 8, 12}) {
      g53cpu_opts opts;
      opts.ncpu = nt;
      auto *p = g53cpu_create(&mt.t, opts);
      for (int c = 0; c < (full ? 4 : 2); c++) {
        for (int i = 0; i < D; i++)
          x[i] = c == 0   ? 0
                 : c == 1 ? value((i & 1 ? 0x80000000 : 0) | 1)
                 : c == 2 ? float(int(rng() % 2001) - 1000) * .001f
                          : float(int(rng() % 2001) - 1000);
        compute(w, x.data(), ids, mt.t, expected);
        g53cpu_run(p, x.data(), es, 8);
        equal(gg.data(), expected[0].data(), 8 * F, "gate");
        equal(uu.data(), expected[1].data(), 8 * F, "up");
        equal(hh.data(), expected[2].data(), 8 * F, "hidden");
        equal(out.data(), expected[3].data(), 8 * D, "down");
      }
      // Benchmark finite, nonzero uniform [-1,1] activations, rather than the
      // preceding overflow case.
      if (full)
        for (auto &v : x)
          v = float(int(rng() % 2001) - 1000) * .001f;
      // Warm every request arena/worker and verify no allocator activity, then
      // measure faults.
      g53cpu_run(p, x.data(), es, 8);
      rusage before{}, after{};
      getrusage(RUSAGE_SELF, &before);
      allocs = 0;
      counting = true;
      for (int i = 0; i < 3; i++)
        g53cpu_run(p, x.data(), es, 8);
      counting = false;
      getrusage(RUSAGE_SELF, &after);
      require(allocs == 0, "request allocations");
      require(after.ru_minflt == before.ru_minflt &&
                  after.ru_majflt == before.ru_majflt,
              "request faults");
      std::cout << "PASS L" << layer << " threads=" << nt
                << " g/u/h/dn; allocations=" << allocs
                << " warm minor/major faults="
                << after.ru_minflt - before.ru_minflt << "/"
                << after.ru_majflt - before.ru_majflt << "\n";
      if (full)
        for (int k : {1, 4, 8}) {
          std::vector<double> times;
          for (int it = 0; it < 7; it++) {
            auto start = std::chrono::steady_clock::now();
            g53cpu_run(p, x.data(), es, k);
            times.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - start)
                                .count() /
                            k);
          }
          std::sort(times.begin(), times.end());
          double fast = times[3];
          memcpy(ref.data(), out.data(), size_t(k) * D * 4);
          std::vector<double> slow;
          for (int it = 0; it < 3; it++) {
            auto start = std::chrono::steady_clock::now();
            g53cpu_set_scalar(p, true);
            g53cpu_run(p, x.data(), es, k);
            g53cpu_set_scalar(p, false);
            slow.push_back(std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - start)
                               .count() /
                           k);
          }
          equal(out.data(), ref.data(), size_t(k) * D, "bench scalar vs fast");
          std::sort(slow.begin(), slow.end());
          std::cout << "BENCH L" << layer << " threads=" << nt << " k=" << k
                    << " fast_us/expert=" << fast
                    << " scalar_us/expert=" << slow[1]
                    << " ratio=" << slow[1] / fast << "\n";
        }
      g53cpu_destroy(p);
    }
  }
  std::cout << "PASS ALL GLM CPU exact tests\n";
  return 0;
} catch (const std::exception &e) {
  counting = false;
  std::cerr << "FAIL " << e.what() << "\n";
  return 1;
}
