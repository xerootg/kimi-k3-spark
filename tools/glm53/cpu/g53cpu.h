// Exact GLM-5.3 CPU routed experts, raw GGUF gate|up|down records.
#pragma once
#include "../g53exact.h"
#include <cstddef>
#include <cstdint>
using g53_mufu_tabs = mufu_tables;
enum { G53CPU_GGUF = 0, G53CPU_KMAX = 16 };
struct g53cpu_expert {
  const uint8_t *rec;
  int tg, tu, td;
  float *out;
  float *g = nullptr, *u = nullptr, *h = nullptr;
  int layout = G53CPU_GGUF;
};
struct g53cpu_opts {
  const int *cpus = nullptr;
  int ncpu = 4;
  // Idle spin time in microseconds; negative spins forever, zero sleeps on
  // futex.
  int spin_us = 0;
  // Reserved K3 options: lookup and interleave currently ignored.
  int lookup = 0;
  bool interleave = false;
  int rt_prio = 0;
};
struct g53cpu_pool;
struct g53cpu_times {
  uint64_t x_done, gu_done[16], dn_done[16];
};
// Maps and prefaults verified raw tables; mappings live until process exit.
bool g53cpu_load_mufu(const char *, g53_mufu_tabs *);
g53cpu_pool *g53cpu_create(const g53_mufu_tabs *, const g53cpu_opts &);
void g53cpu_destroy(g53cpu_pool *);
// Synchronous two-phase submit. No allocation. One caller per pool.
void g53cpu_submit(g53cpu_pool *, const float *, const g53cpu_expert *, int);
void g53cpu_run(g53cpu_pool *, const float *, const g53cpu_expert *, int);
bool g53cpu_done(g53cpu_pool *, int);
void g53cpu_wait(g53cpu_pool *, int);
void g53cpu_wait_all(g53cpu_pool *);
// Phase timestamps are currently unavailable (returns zeros).
void g53cpu_last_times(g53cpu_pool *, g53cpu_times *);
size_t g53cpu_rec_bytes(int, int, int, int = G53CPU_GGUF);
float g53cpu_row(int, const uint8_t *, int, const g53exact::Q8 *);
void g53cpu_reference(const uint8_t *, int, int, int, const float *,
                      const g53_mufu_tabs &, float *);
// Benchmark selection; set only while pool is idle.
void g53cpu_set_scalar(g53cpu_pool *, bool);
