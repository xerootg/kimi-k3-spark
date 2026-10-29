# CPU expert kernels: k3ref (scalar oracle) and k3cpu (AVX2 + AVX-VNNI library)

- `k3ref.cpp`: the scalar oracle of `../SPEC.md` (see `../README.md`). Slow (6.8 ms per expert), only for checking.
- `k3cpu.h` / `k3cpu.cpp`: a library that computes one decode token's routed experts (q8_1 of x, gate/up MMVQ, SiTU,
  q8_1 of h, down MMVQ) bit-identical to the Spark's CUDA rows, fast on Intel hybrid CPUs (Arrow Lake: AVX2, AVX-VNNI,
  BMI2, F16C, no AVX-512). `k3cpu_test.cpp` is its validation and benchmark driver; `build_k3cpu.sh` builds it static.

## API (k3cpu.h)

```
k3_mufu_tabs mt;  k3cpu_load_mufu("<dir with rcp/ex2p/ex2n/tanh.bin>", &mt);   // packs like k3x: 0.75 GB, 2 MiB pages
k3cpu_opts o;  o.cpus = cpus; o.ncpu = n;                                       // one worker per cpu, pinned
k3cpu_pool * P = k3cpu_create(&mt, o);                                         // threads created once
k3cpu_expert e[k] = {{rec, tg, tu, td, out_row, nullptr, nullptr, nullptr}, ...};
k3cpu_submit(P, x, e, k);   k3cpu_wait(P, 0); ...   // or k3cpu_run(P, x, e, k)
```

- `rec` points at a record in any arena; nothing is copied. `layout` says which: `K3CPU_GGUF` (gate | up | down in the
  GGUF row layout of `../FORMAT.md`, the remote-set `.bin` files) or `K3CPU_V7` (each matrix repacked as k3x's v7 GPU
  layout, `../sycl/k3x.cpp` `v7_off`; the layout of S1's shared host-USM arena), so the CPU and the iGPU can work on one
  copy. Types: 19 = IQ1_S, 16 = IQ2_XXS. `out` gets the 3584-float unweighted down row; `g`/`u`/`h` (optional) the
  intermediate stages.
- No allocation per call. A token is 8 + 208 k tasks (q8_1 of x in 8 pieces, gate/up in chunks of 32 rows, down in
  chunks of 32 rows) claimed dynamically by the workers, so P- and E-cores share the work at their own speed. The down
  tasks of an expert wait for its gate/up tasks. `k3cpu_submit` returns after ~0.1 us; the caller polls `k3cpu_done`/`k3cpu_wait`
  per expert (stream rows as they finish), `interleave` orders tasks expert by expert instead of all gate/up first.
- The packed MUFU table format is k3x's `k3_ptab` (`../sycl/k3x_math.h`), so a `k3_mufu_tabs` pointing at tables that
  another component already holds can be passed in directly.
- `k3cpu_opts.rt_prio > 0` runs the workers SCHED_FIFO (needs CAP_SYS_NICE, else a warning); keep `spin_us` short then.
- In the k3x server: `K3X_CPU_CPUS` and the rest of the CPU-half knobs in `../sycl/README.md`. On nuc2 (dual channel)
  11 workers on cpus 1-11 compute one v7 expert in ~0.21-0.25 ms; 8 E-cores alone ~0.34 ms, about nuc3's single-channel
  speed, so the kernels are compute-bound there.

## How it stays bit-exact

Per output row the float arithmetic is SPEC section 3 in CUDA's order: `p = fma(dW, fma(dsx, sumi, dsy*delta), +0)` per
virtual thread (8 sub-blocks of a 256-block = 8 ymm lanes), then `((w0 + w1) + w2) + w3` per lane and the xor
butterfly as a tree. Workers run with MXCSR DAZ|FTZ, which is CUDA's `.FTZ` on every FADD/FMUL/FFMA (x86 detects tininess
after rounding); every FMA is an explicit `_mm256_fmadd_ps`; built with `-ffp-contract=off -fno-fast-math`. Only the
integer dot products are reorganised (exact): the activations are stored transposed (`qt[kb][r][j]` = dword r of
q8_1 block 8kb+j) so `vpdpbusd` lane j accumulates sub-block j directly. IQ1_S grid values are 0..2, so they are the
unsigned operand of `vpdpbusd` as they are; the high nibbles are used in place (values 0/16/32) and the sum shifted
right by 4. IQ2_XXS signs are applied to the activations with `vpsignb` (|q| <= 127 always). q8_1 uses the vector path
when everything is finite and d is normal, and the literal SPEC path otherwise (NaN/inf, d flushed to zero). SiTU is
`k3_act` from `../sycl/k3x_act.h` (the same code as k3x's host reference). `_cvtsh_ss` under DAZ is checked against the
exact half conversion for all 65536 inputs at pool creation.

## Grid lookups: P-cores gather, E-cores load

IQ1_S needs 32 lookups into the 2048-entry grid per 256 weights. Measured on nuc3 (4 x vpgatherdd ymm = 32 lookups):
3.9-5.0 ns on a Lion Cove P-core, 10.4 ns on a Skymont E-core, where 32 scalar loads take 6.9 ns. So each worker reads
its core type (CPUID 0x1A) and the E-cores use a scalar-load kernel: indices built with BMI2 `pdep` (all four 11-bit indices
of a sub-block in 3 instructions), 64-bit entries of a nibble-expanded grid (`g64`), four per register against an
activation copy in matching order (`q64`), lane pairs added at the end. IQ2_XXS on E-cores: magnitudes from
`iq2xxs_grid` and the 8 sign multipliers from a 128-entry table (parity included). `k3cpu_opts.lookup` forces either.

Per core, one expert, records in L3 (nuc3, ms per expert): P-core gather 1.01 (IQ1_S layer) / 1.45 (IQ2_XXS gate/up
layer); E-core scalar 1.44 / 1.90 (gather on the E-core: 2.46 / 3.81). Before the pairing, inlining and pdep work the same
numbers were P 1.45, E 3.1.

Other things that mattered (each measured):
- rows go in pairs (gate i with up i, down r with r+1) so a block's activation loads serve two rows; the 14-block loop is
  kept rolled: fully unrolled, GCC hoisted all 3.5 KB of activation loads and spilled them to the stack for every row.
- the packed MUFU tables are random-access over 0.75 GB: they live in 2 MiB transparent huge pages (one TLB miss less
  per lookup), and the lookups are prefetched one row ahead (tanh(g), tanh(u), ex2(g), then rcp(ex2+1) once ex2 is cached)
  with SiTU computed two rows behind the dot products. Before: up to 35 % of a P-core's time was table misses.
- q8_1(x) is done by the workers (8 tasks, rcp lines prefetched per block), not the caller: posting a token takes
  0.1 us instead of 9 us.
- software prefetch of weight rows: no gain (hardware prefetch suffices), left off (`K3CPU_PF_ROWS`).

## v7 records

In v7 a 16-byte load is the same dword of 4 consecutive rows (blocks padded to 13 / 17 dwords, dword stride 64 bytes),
so the v7 kernels do 4 rows per pass with ymm lanes = (sub-block 2sp or 2sp+1, row). The CUDA reduction tree then needs
no shuffles: with S[m][sp] = ((p(m) + p(m+4)) + p(m+8)) + p(m+12) per lane, the xor levels 16, 8, 4, 2 are lane-wise adds
of whole registers (m with m+2, m 0 with 1, sp with sp+2, sp 0 with 1) and level 1 adds the two 128-bit halves, giving
the 4 rows' results. The 2-byte offset of the block payload (after the half d) costs a 16-bit shift per dword. The x
side uses a third copy of the activations (`qv`, each dword repeated for the 4 rows); the E-core kernel broadcasts its
8 bytes per row from `q64` instead. Bit-exact like the GGUF path (all dumps, both lookup kernels, repacked in memory
by k3x's own `v7_repack_matrix`, `K3B_V7=1`). It is slower per core than GGUF: P-core 1.24 vs 1.00 ms per expert,
E-core 1.95 vs 1.42 (more index work per lane, the shifts, and more L1 pressure on the 32 KB E-core L1).

## Results (2026-10-29, nuc3: Core Ultra 5 225H, one DDR5-5600 channel)

Validation (`k3cpu_test validate`, A's dumps): L1 (IQ1_S x3), L11 (IQ2_XXS gate/up, IQ1_S down), L83 (IQ1_S gate/up,
IQ2_XXS down): g, u, h and dn of every case (256 / 160 / 96 cases x 16 experts) bit-identical, 0 mismatches, for both
record layouts (GGUF, v7) x the gather kernel, the scalar-load kernel and the automatic mix (18 runs on cpus 0-11, binary
md5 c6f629b3..., plus earlier runs on P-cores only and E-cores only).
`k3cpu_test check` (random and edge-case tokens incl. denormals, 1e-38 and 1e4 spikes, vs k3x_math.h): 0 differences.

Speed, one token, back to back, records cold (round robin over 32 / 16 records), x from the dumps, caller pinned to
cpu 12 (an LP-E core with no worker), median ms per token and per expert:

| layout | cores | layer | k=1 | k=2 | k=3 | k=4 | per expert (k=4) | weights |
|---|---|---|---|---|---|---|---|---|
| GGUF | 8 E (4-11) | L1 IQ1_S | 0.253 | 0.497 | 0.738 | 0.978 | 0.244 | 26 GB/s |
| GGUF | 8 E (4-11) | L11 IQ2_XXS gate/up | 0.282 | 0.549 | 0.823 | 1.098 | 0.275 | 29 GB/s |
| GGUF | 8 E (4-11) | L83 IQ2_XXS down | 0.268 | 0.521 | 0.782 | 1.044 | 0.261 | 27 GB/s |
| GGUF | 12 (0-11) | L1 | 0.223 | 0.426 | 0.639 | 0.847 | 0.212 | 31 GB/s |
| GGUF | 12 (0-11) | L11 | 0.247 | 0.472 | 0.705 | 0.964 | 0.241 | 33 GB/s |
| GGUF | 12 (0-11) | L83 | 0.238 | 0.453 | 0.681 | 0.909 | 0.227 | 31 GB/s |
| GGUF | 4 P (0-3) | L1 | 0.317 | 0.624 | 0.934 | 1.269 | 0.317 | 20 GB/s |
| v7 | 8 E (4-11) | L1 | 0.340 | 0.659 | 0.990 | 1.320 | 0.330 | 20 GB/s |
| v7 | 8 E (4-11) | L11 | 0.396 | 0.772 | 1.154 | 1.549 | 0.387 | 21 GB/s |
| v7 | 8 E (4-11) | L83 | 0.363 | 0.711 | 1.064 | 1.416 | 0.354 | 21 GB/s |
| v7 | 12 (0-11) | L1 | 0.271 | 0.506 | 0.758 | 1.036 | 0.259 | 26 GB/s |
| v7 | 12 (0-11) | L11 | 0.324 | 0.611 | 0.925 | 1.264 | 0.316 | 26 GB/s |
| v7 | 12 (0-11) | L83 | 0.291 | 0.545 | 0.813 | 1.117 | 0.279 | 27 GB/s |
| v7 | 4 P (0-3) | L1 | 0.401 | 0.786 | (1.82) | (2.44) | | |

(The v7 P-core k=3/4 values in brackets were disturbed, probably by S1's iGPU runs on cpus 0-3 of nuc3 at the time.)
Not measured on nuc2: its k3x server was serving the Spark (8-20 MB/s) whenever I checked, and the task allowed a nuc2
run only while it is idle. nuc2 has two memory channels, so its E-cores should be compute-bound near the per-core
numbers above (8 x 1.42 ms -> ~0.18-0.2 ms per expert for GGUF).

For comparison, the nuc2 iGPU (k3x, M's trace): 0.78 ms warm for the engine's k mix (1.69 experts), 1.04 ms after an
idle gap; ggml's own CPU kernels 0.48 ms at k=1 on nuc2; k3ref fast path 6.8 ms.

P-cores versus E-cores on this chip:
- one P-core streams only 8.9 GB/s from DRAM, one E-core 26 GB/s, the 8 E-cores 40 GB/s (single-channel peak 44.8), the
  4 P-cores 24 GB/s. The E-cores are the better workers here: 8 E-cores alone are faster than the 4 P-cores, and
  adding the P-cores to the E-cores helps only at k >= 2.
- with 8 E-cores busy, their clock drops from 4.29 to 3.7-4.0 GHz and IPC by ~10 %, so 8 E-cores reach ~5.7x one
  E-core, not 8x. The remaining gap to the 40 GB/s roof is compute, mostly the grid lookups.
- a P-core worker pinned on the same core as other busy threads (kubelet, containerd) gets preempted for whole
  scheduler slices: p90 of 15-25 ms in early runs. Pin the caller to a core without a worker.

Idle gaps (engine pacing, one token per layer every 3-8 ms):
- workers must spin between tokens (`spin_us`, default 20 ms, then futex sleep): sleeping workers cost 0.5-0.8 ms per
  token (wake-up from C-states, staggered).
- even with spinning workers, after a 5 ms gap the token time is bimodal on nuc3: a variable share of tokens (25-90 % per
  run) runs ~1.7x slower in every phase (k=1 0.43 ms instead of 0.26), at unchanged E-core clocks (4.25 GHz, perf).
  It got worse with a busy P-core (a spinning shell on cpu 1, or the caller spinning) and did not go away with the
  uncore clock pinned to its maximum, with a non-pause spin, or with a light memory touch while spinning. Mechanism
  not found; probably memory or fabric power state, like M's slower-after-idle GPU kernels. S1 was running iGPU bursts
  on nuc3 during these runs (a possible confounder). To be measured on nuc2 under the real server.

## Build and run

```
./build_k3cpu.sh                       # g++ -O2 -march=x86-64-v3 -mavxvnni -ffp-contract=off -fno-fast-math -static
k3cpu_test validate MUFU_DIR RECORDS N TG TU TD DUMP_DIR CPUS [ncases]
k3cpu_test check    MUFU_DIR RECORDS N TG TU TD CPUS [ntokens]
k3cpu_test bench    MUFU_DIR RECORDS N TG TU TD CPUS ITERS [DUMP_DIR]
# any mode: K3B_V7=1 repacks the records to v7 in memory and runs the v7 kernels
#   e.g. R=~/k3-bitexact/ref; taskset -c 12 ./k3cpu_test bench $R/mufu $R/rec_L1.bin 32 19 19 19 4-11 300 $R/dumps/L1
# bench knobs: K3B_KS=1,2 (k values), K3B_GAP_US (sleep between tokens), K3B_GAP_SPIN=1, K3B_SPIN_US (worker spin),
#   K3B_LOOKUP=1|2 (gather | scalar everywhere), K3B_INTERLEAVE=1, K3B_TRACE=file (per token: k it x ms gu dn posted CLOCK_MONOTONIC-ns), K3B_RT=<prio> (SCHED_FIFO workers)
# library knobs: K3CPU_PF_ROWS (weight prefetch distance), K3CPU_SPIN_MODE (0 pause, 1 no pause, 2 pause + memory touch)
```

## k3cs: the CPU-only server

`k3cs IDX BIN PORT` (`k3cs.cpp`, `build_k3cs.sh`, also in the k3x-serve image as `/opt/k3x/k3cs`) serves k3x serveidx's wire
protocol from k3cpu alone: no SYCL and no iGPU, records kept in the GGUF layout (k3cpu is 20-27 % faster per expert on GGUF
than on v7, and plain memory has no i915 residency limit), in 1 GiB transparent-huge-page chunks locked with mlockall once
loaded. Knobs as k3x where the meaning is the same (`K3X_MUFU`, `K3X_MAX_GB`, `K3X_SETOUT`, `K3X_FULLCHECK`, `K3X_CPU_*`,
`K3X_RT`, `K3X_RX_SPIN_US`, `K3X_MEMWARM*`, `K3X_TRACE*`), plus `K3X_MAIN_CPU` (pin the request thread once serving; do not
start it under taskset, FULLCHECK's reference threads use every cpu). `K3X_FULLCHECK` runs every served record through the
serving kernels and compares it bit for bit with the scalar host reference of `../sycl/k3x_math.h`; an all-zero row is sent
as NaN. Checked on nuc3: FULLCHECK 636/636 and remotecanary L1/L11/L83 at K = 1, 2, 3, 4, 16 bit-identical; 21 GB (3117
records) loads and locks on the 32 GB box with 7 GB left. nuc3 (one channel), RT on, 5 ms gaps, server time p50 per k:
0.39 / 0.61 / 0.84 / 1.07 ms for k = 1..4.
