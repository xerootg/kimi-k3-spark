// k3cpu.h -- bit-exact Kimi K3 routed-expert FFN for one decode token on x86 (AVX2 + AVX-VNNI), as a library.
//
// Reproduces the Spark's ggml-cuda rows (../SPEC.md) bit for bit: q8_1(x), gate/up MMVQ, SiTU with the tabulated
// MUFU.TANH/EX2/RCP, q8_1(h), down MMVQ. Per row the arithmetic is SPEC section 3 in CUDA's order; only the integer
// dot products are reorganised (exact). See README.md in this directory for the design, numbers and P/E-core notes.
//
// Records are in the GGUF row layout (gate rows | up rows | down rows, the layout of ../FORMAT.md and of the remote-set
// .bin files): a pointer into any shared arena works, nothing is copied. Threads are created once by k3cpu_create and
// pinned; a call allocates nothing. Not thread-safe per pool: one caller submits at a time.
#pragma once
#include <cstddef>
#include <cstdint>
#include "../sycl/k3x_math.h"   // k3_mufu_tabs / k3_ptab: the same packed MUFU tables as the k3x SYCL server

enum { K3CPU_T_IQ2_XXS = 16, K3CPU_T_IQ1_S = 19, K3CPU_KMAX = 16 };

// Load ../mufu raw tables ({rcp,ex2p,ex2n,tanh}.bin in dir) and pack them exactly like k3x (blocks of 64 entries,
// int8 residuals, escapes; ~0.75 GB in all, verified entry by entry). The memory stays owned by the library.
// Returns false (with a message on stderr) if a table is missing or fails verification.
bool k3cpu_load_mufu(const char * dir, k3_mufu_tabs * out);
// A k3_mufu_tabs whose pointers come from elsewhere (e.g. the k3x server's packed tables in host-visible USM) can be
// passed to k3cpu_create directly: the format is identical.

struct k3cpu_expert {
    const uint8_t * rec;      // gate | up | down, GGUF row layout
    int tg, tu, td;           // ggml types: 19 = IQ1_S, 16 = IQ2_XXS
    float * out;              // 3584 floats: the unweighted down row (what the remote returns)
    float * g, * u, * h;      // optional (nullptr): 3072 floats each, for validation
};

struct k3cpu_pool;
struct k3cpu_opts {
    const int * cpus = nullptr; int ncpu = 0;   // worker i pinned to cpus[i]; ncpu workers (no pinning if cpus == nullptr)
    int spin_us = 20000;      // idle workers spin this long (pause loop) before sleeping on a futex; -1 = spin forever
    int lookup = 0;           // IQ1_S grid lookup per worker: 0 = by core type (P: vpgatherdd, E: scalar loads),
                              // 1 = vpgatherdd everywhere, 2 = scalar loads everywhere
    bool interleave = false;  // task order gu(e0) dn(e0) gu(e1) ..: expert 0 finishes first (streaming), default batches
                              // gu(e0) gu(e1) .. dn(e0) dn(e1) .. (fewer stalls, earliest completion of the whole token)
};
k3cpu_pool * k3cpu_create(const k3_mufu_tabs * mt, const k3cpu_opts & o);
void k3cpu_destroy(k3cpu_pool * p);

// Asynchronous: posts the token to the workers (q8_1 of x is their first 8 tasks) and returns (~1 us).
// x: 3584 floats; x and e[i].out/g/u/h must stay valid until done; e[0..k-1] are copied. 1 <= k <= K3CPU_KMAX.
void k3cpu_submit(k3cpu_pool * p, const float * x, const k3cpu_expert * e, int k);
bool k3cpu_done(k3cpu_pool * p, int e);    // expert e's out row complete (non-blocking)
void k3cpu_wait(k3cpu_pool * p, int e);    // spin until expert e's out row is complete
void k3cpu_wait_all(k3cpu_pool * p);
// Synchronous convenience.
void k3cpu_run(k3cpu_pool * p, const float * x, const k3cpu_expert * e, int k);

// Per-phase timing of the last job (ns, CLOCK_MONOTONIC_RAW, relative to submit): job posted, each
// expert's gate/up done, each expert done.
struct k3cpu_times { uint64_t x_done, gu_done[K3CPU_KMAX], dn_done[K3CPU_KMAX]; };   // x_done: job posted
void k3cpu_last_times(k3cpu_pool * p, k3cpu_times * t);

size_t k3cpu_rec_bytes(int tg, int tu, int td);
