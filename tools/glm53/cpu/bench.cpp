// Offline compute-only benchmark: random grids, nonzero quantized activations,
// zero gate.
#include "g53cpu.h"
#include "mapped_fixture.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <random>
#include <sstream>
#include <string>
#include <sys/prctl.h>
#include <vector>
#ifdef G53_BENCH_PAIRED
struct baseline_pool;
baseline_pool *baseline_create(const g53_mufu_tabs *, const g53cpu_opts &);
void baseline_run(baseline_pool *, const float *, const g53cpu_expert *, int);
void baseline_destroy(baseline_pool *);
#endif
int main(int argc, char **argv) {
  if (prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0))
    return 2;
  int nt = argc > 1 ? atoi(argv[1]) : 8, reps = argc > 2 ? atoi(argv[2]) : 15;
  if (nt < 1 || nt > 256 || reps < 1)
    return 2;
  g53_mufu_tabs mt{};
  g53cpu_opts opts;
  opts.ncpu = nt;
  std::vector<int> cpus;
  if (const char *list = getenv("G53_BENCH_CPUS")) {
    std::stringstream stream(list);
    std::string item;
    while (std::getline(stream, item, ','))
      cpus.push_back(std::stoi(item));
    if (int(cpus.size()) != nt)
      return 2;
    opts.cpus = cpus.data();
  }
  if (const char *caller = getenv("G53_BENCH_CALLER")) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(atoi(caller), &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof mask, &mask))
      return 2;
  }
  auto *p = g53cpu_create(&mt, opts);
  if (!p)
    return 2;
#ifdef G53_BENCH_PAIRED
  auto *bp = baseline_create(&mt, opts);
#endif
  std::mt19937 rng(178);
  for (int layer : {3, 8, 75}) {
    int tg = layer == 8 ? 18 : 17, td = layer == 3 ? 18 : 23;
    size_t rs = g53cpu_rec_bytes(tg, tg, td);
    std::vector<uint8_t> raw(rs * 8);
    for (auto &v : raw)
      v = rng();
    // Zero gate and power-of-two x scale keep MUFU in its closed-form subset.
    for (int e = 0; e < 8; e++) {
      size_t off = rs * e;
      for (int ty : {tg, tg, td}) {
        for (int b = 0; b < 49152; b++) {
          raw[off] = 0;
          raw[off + 1] = off < rs * e + size_t(49152) * g53exact::blockbytes(tg)
                             ? 0
                             : 0x30;
          off += g53exact::blockbytes(ty);
        }
      }
    }
    MappedFixture rec(raw);
    std::vector<float> x(6144), out(8 * 6144);
    for (int i = 0; i < 6144; ++i)
      x[i] = (i % 32 == 0 ? 127.f : float(i % 255 - 127)) * .0078125f;
    g53cpu_expert es[8];
    for (int e = 0; e < 8; e++)
      es[e] = {rec.data() + e * rs, tg, tg, td, out.data() + e * 6144};
    for (int k : {1, 4, 8}) {
      for (int w = 0; w < 3; w++)
        g53cpu_run(p, x.data(), es, k);
#ifdef G53_BENCH_PAIRED
      for (int w = 0; w < 3; w++)
        baseline_run(bp, x.data(), es, k);
      std::vector<double> before;
#endif
      std::vector<double> times;
      for (int it = 0; it < reps; it++) {
        for (int order = 0; order < 2; order++) {
#ifdef G53_BENCH_PAIRED
          bool base = ((it + order) & 1) == 0;
#else
          if (order)
            continue;
#endif
          auto start = std::chrono::steady_clock::now();
#ifdef G53_BENCH_PAIRED
          if (base)
            baseline_run(bp, x.data(), es, k);
          else
#endif
            g53cpu_run(p, x.data(), es, k);
          double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - start)
                          .count() /
                      k;
#ifdef G53_BENCH_PAIRED
          if (base)
            before.push_back(us);
          else
#endif
            times.push_back(us);
        }
      }
#ifdef G53_BENCH_PAIRED
      std::sort(before.begin(), before.end());
      printf("BEFORE L%d threads=%d k=%d us/expert=%.2f\n", layer, nt, k,
             before[before.size() / 2]);
#endif
      std::sort(times.begin(), times.end());
      printf("BENCH L%d threads=%d k=%d us/expert=%.2f\n", layer, nt, k,
             times[times.size() / 2]);
      fflush(stdout);
    }
  }
  g53cpu_destroy(p);
#ifdef G53_BENCH_PAIRED
  baseline_destroy(bp);
#endif
}
