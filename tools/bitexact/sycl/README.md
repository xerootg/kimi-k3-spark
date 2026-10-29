# k3x: bit-exact routed-expert FFN on the Intel Arc iGPU (SYCL)

Standalone SYCL implementation of one Kimi K3 decode token's routed-expert FFN (q8_1 of x, gate/up MMVQ,
SiTU, q8_1 of h, down MMVQ) that reproduces the Spark's production CUDA kernels bit for bit, following
`../SPEC.md` (FTZ on every FADD/FMUL/FFMA, explicit FMAs, CUDA's 4-warp reduction order, MUFU.RCP/EX2/TANH from
the exhaustive tables in `../mufu/`). Not built on ggml-sycl.

- `k3x_math.h`: shared host/device arithmetic (q8_1, IQ1_S / IQ2_XXS vec_dot tails, row reduction, FTZ helpers).
- `k3x_mufu.h`: MUFU emulation from the tables (same layout and edge rules as `../mufu/mufu_emu.h`).
- `k3x_act.h`: the SiTU chain (SPEC section 4).
- `k3x.cpp`: device kernels, host reference, and the modes below. Current matvec ("v7"): records are repacked at
  load so 4 lanes share each output row (lane = kb0 group, CUDA's first two xor-tree levels as lane shuffles),
  IGC's native DP4A (`__builtin_IB_dp4a_ss`; `dot_acc` compiles to 32-bit multiply emulation on this iGPU), grid
  tables and activation in SLM, 512-item work-groups. Older variants stay behind macros (MV2..MV4, V5, V6).
- `build.sh`: build recipe (oneAPI 2026.1.1 container, `-ffp-model=precise -ffp-contract=off -fno-fast-math`).

Run on nuc2 from a directory holding `k3x` and the runtime `lib/` (see `build.sh`), with the tables in
`K3X_MUFU` (default `../ref/mufu`), as `sudo -u xero -g render env LD_LIBRARY_PATH=lib ./k3x ...`:

```
k3x RECORDS N TG TD validate DUMPDIR [ncases]   # vs agent A's dumps (x/ids/g/u/h/dn), counts per stage
k3x RECORDS N TG TD check [n]                   # device vs host reference on random inputs ("synthetic" records ok)
k3x RECORDS N TG TD bench k1,k2,.. ITERS        # p50/p90 per token (K3X_PROF=1: per-kernel times)
k3x RECORDS N TG TD serve PORT                  # tools/distributed/rrclient.c protocol, ids = record slots
k3x IDX BIN - - serveidx PORT                   # remote-set server: IDX "<gid> <tg> <td> <offset>", BIN records
```

Serving: `serve`/`serveidx` busy-poll the socket and stream each expert's row as soon as its down matmul is done
(`K3X_NOSTREAM=1`: one write). `serveidx` caps the GPU-resident records at `K3X_MAX_GB` (default 24), writes the served
(layer, expert) list to `K3X_SETOUT`, self-checks one token at startup and sends NaN for any all-zero row: above
roughly 34-42 GB resident on the Arc 130T, kernel submissions sometimes fail with no error and leave rows zero.

Zero-copy request path (S1, 2026-10-29; defaults on, arithmetic unchanged, every mode validated bit for bit):
- **One record arena in host USM** (`K3X_ARENA=host`, default; `device` = the old device-USM chunks). `serveidx` repacks
  each record once, straight into 1 GiB `sycl::malloc_host` chunks; the iGPU and CPU code read the same bytes. Layout and
  `K3X_ARENA_MAP=<file>` (per record: gid, types, chunk, offset, bytes) are documented at `struct K3xArena` in `k3x.cpp`.
  The runtime maps host USM as an i915 GEM object, write-back, 4 KiB CPU pages (no 2 MiB pages); CPU reads run at
  malloc speed. The full set (7031 records incl. the 64 canary records, 48.3 GB + 0.75 GB tables) passed `K3X_FULLCHECK`
  on 3/3 cold starts on nuc2 with this arena. GPU speed host vs device USM: equal within ~3 % (nuc2 bench).
- **Request slot in host USM**: header and x are `recv()`d straight into it, the q8_1 kernel reads x there, the down
  kernels write the rows there, and `send()` sends from there. No H2D/D2H copies, no `std::vector`/`std::function`/`new`
  on the request path in k3x itself (the SYCL runtime still allocates ~20 times per submission: `mcount.c` counts it).
- **Record pointers from a host-USM table** read by the kernels at run time (`recs[]`), not a by-value kernel argument:
  makes pre-recorded graphs possible and is faster (down per expert 122 -> 93-101 us). A select between the two kinds of
  pointer in one kernel made every matvec ~2x slower (IGC codegen): never mix them.
- **Immediate command lists** (`UR_L0_USE_IMMEDIATE_COMMANDLISTS=1`, set by `main` unless already set). With the default
  batched lists nothing reaches the GPU until an event is queried. The level_zero v2 adapter returned stale host-USM data.
- Host USM on this iGPU is coherent only at command boundaries: CPU stores are not seen by a running kernel (even system-
  scope atomics; a GPU spin/mailbox kernel cannot work, and a 3 s spinning kernel is a GPU hang), and in-kernel
  completion flags (fence + atomic counter) gave stale rows. So the CPU writes x/ids before the submission and reads a
  row after its command's event completed (polled).
- Options: `K3X_DOWN_SPLIT=0` one down launch for all k (rows arrive together), `K3X_GRAPH=1` the per-k sequence as a
  SYCL graph (limited graphs only on this runtime; one submission), `K3X_FUSE_QX_RT=1` q8_1(x) inside gate/up.
  `K3X_CTL=<file>` switches `zc= split= graph= qx= warm=` per connection without a reload (A/B). Measured on nuc2 these
  are within ~±30 us of the default (split streaming). Gate/up + SiTU in one kernel was 10-25 % slower: removed.
- **Keep-awake while a client is connected**: `K3X_FORCEWAKE=<path>` holds
  `/sys/kernel/debug/dri/0/i915_forcewake_user` open from accept() to close() (needs root and debugfs in the
  container), and `K3X_MEMWARM=<MB/s>` streams a 64 MiB buffer on a spare core while waiting for a request (the memory
  clocks drop between requests). 4000 MB/s was best of 1000/2000/4000/8000.

nuc2, full set, rrclient from the Spark (`RR_IDX` real ids, dist = 31 % k=1 / 69 % k=2), p50 / p90 ms:

| server | k=1 | k=2 | k=3 | dist | dist, 8 ms gap | gap + forcewake | gap + fw + memwarm 4000 |
|---|---|---|---|---|---|---|---|
| before (65aaeb0 = production v11 path) | 0.812 | 1.081 | 1.347 | 1.072 / 1.207 | 2.025 / 2.866 | 1.310 / 1.490 | |
| zero copy, split streaming (default) | 0.808 | 1.007 | 1.256 | 0.987 / 1.133 | 1.835 / 2.516 | 1.283 / 1.407 | 1.028 / 1.226 |
| zero copy, graph + one down launch | 0.753 | 0.982 | 1.222 | 0.958 / 1.046 | 1.836 / 2.470 | 1.217 / 1.352 | 0.997 / 1.182 |

Server stages, zero copy, warm, k=2 (K3X_TRACE_PROF, us p50): x received 60 after the first byte; GPU q8_1(x) 29,
gate/up 279, SiTU + q8_1(h) 95, down 93 + 93; GPU gaps between commands 14-42 (one per boundary); all submissions
done 138 after x; first byte -> last row sent 841. After an 8 ms idle without keep-awake: submissions 434 (RC6 exit),
gate/up 374, SiTU 151, down 134 each; with forcewake + memwarm the kernels are back at 306 / 96 / 101.

CPU half (I, 2026-10-29; off unless `K3X_CPU_CPUS` is set): the server also holds a k3cpu pool (`../cpu/k3cpu.h`, S3's
bit-exact AVX2 + AVX-VNNI kernels) that reads the same v7 records in place from the host-USM arena, so there is still one
copy of the experts. Per request the first n experts (slots 0..n-1) go to the CPU and the rest to the iGPU, started
together; rows go out in slot order (the CPU's first), so the wire format is unchanged.
- `K3X_CPU_CPUS=<list>` worker cpus (one pinned worker each), `K3X_CPU_SPLIT=<n at k=1>,<n at k=2>,...` experts on the CPU
  per k (the last value repeats, clamped to k; default 1), `K3X_CPU_SPIN_US` (20000), `K3X_CPU_RT=<prio>` SCHED_FIFO workers
  (spin capped at 2 ms), `K3X_CPU_INTERLEAVE`, `K3X_CPU_LOOKUP`. `K3X_MEMWARM_CPU=<cpu>` pins the memwarm thread.
  `K3X_CTL` takes `cpu=<split list>` and `fw=0|1` per connection. The CPU has its own packed MUFU tables (0.75 GB).
- Checks: `K3X_FULLCHECK` runs every served record on the CPU too (vs the host reference); `validate` with `K3X_VCPU=n`
  runs the serving path's split (n experts of each case on the CPU) and checks g/u/h/dn; `bench` with `K3X_BCPU=n`.
- Build: `build.sh` compiles `../cpu/k3cpu.cpp` with g++ (the flags of `../cpu/build_k3cpu.sh`) and links it (mount
  `tools/bitexact` at /w and run `/w/sycl/build.sh`; `NO_CPU=1` leaves it out).

What decides the split on nuc2 (Core Ultra 5 225H, Arc 130T): the CPU computes one expert in ~0.25 ms (11 workers,
cpus 1-11) and does not fall asleep between requests the way the GT does, so for the engine's k = 1-2 the CPU alone is
fastest; at k = 3 two experts on the CPU and one on the iGPU beat either alone. The iGPU after an 8 ms gap is slow even
with forcewake + memwarm. Two host conditions matter: EPP `performance` on nuc2 (with `balance_performance` 40-55 % of
the CPU requests after a gap run ~1.5x slower: the "slow after idle" mode S3 saw; EPP performance removes most of it)
and the memwarm thread off the worker cpus (`K3X_MEMWARM_CPU=12`, an LP-E core). With EPP balance_performance the
memwarm thread on cpu 12 made things worse.

rrclient from the Spark, full set, p50 / p90 ms (dist = 31 % k=1 / 69 % k=2; gap = 8 ms between requests, like the engine):

| server (nuc2) | k=1 | k=2 | k=3 | dist | gap k=1 | gap k=2 | gap dist |
|---|---|---|---|---|---|---|---|
| production g4de37ce (pod) | 0.828 | 1.159 | 1.264 | 1.135 / 1.192 | 1.788 / 2.199 | 2.246 / 2.555 | 2.107 / 2.525 |
| iGPU only (S1 path), forcewake + memwarm 4000 | 0.738 | 1.004 | 1.263 | 0.983 / 1.083 | 0.803 / 0.909 | 1.076 / 1.259 | 1.029 / 1.217 |
| 1 expert CPU (8 E-cores) + rest iGPU, fw + memwarm | 0.543 | 0.807 | 1.134 | 0.798 / 0.883 | 0.708 / 0.765 | 1.062 / 1.249 | 0.988 / 1.240 |
| CPU for k <= 2 (8 E-cores 4-11), no keep-awake | 0.533 | 0.885 | 0.973 | 0.863 / 0.959 | 0.683 / 0.742 | 0.949 / 1.201 | 0.839 / 1.198 |
| same, 11 workers (cpus 1-11, server on cpu 0) | 0.461 | 0.712 | 0.880 | 0.692 / 0.801 | 0.633 / 0.744 | 1.008 / 1.170 | 0.994 / 1.159 |
| same + EPP performance | 0.443 | 0.675 | 0.861 | 0.661 / 0.755 | 0.589 / 0.696 | 0.733 / 1.023 | 0.679 / 0.991 |
| same + EPP performance + memwarm 4000 on cpu 12 (the engine legs' config) | 0.436 | 0.662 | 0.859 | 0.664 / 0.777 | 0.456 / 0.587 | 0.684 / 0.818 | 0.671 / 0.789 |

Server side in the engine leg c2 (that config, K3X_TRACE): first byte -> last row sent k=1 298 / 394 us, k=2 516 /
641 us p50 / p90 (CPU per expert ~0.21-0.25 ms). Tail: p99 1.06 ms, p99.9 3.4 ms, max 90 ms; the long ones are the CPU
workers finishing late (c_dn in the trace), also with SCHED_FIFO workers (`K3X_CPU_RT=50`: still 1-8 per 4000 requests
over 3 ms), and they are not page faults; the cause is open. The engine's breaker tripped 4 and 1 times in the two legs.

Engine A/B (2026-10-29, native on the Spark, production env, 3 x 200 tokens + warm-up per leg, every run gold): tok/s
- a production engine e844a2a + production server: 2.047 / 2.045 / 2.040
- b S2 engine c4efdca + production server: 2.052 / 2.057 / 2.054, 2.040 / 2.052 / 2.048
- c S2 engine + this server (CPU for k <= 2): 2.102 / 2.101 / 2.088, 2.111 / 2.111 / 2.106
- d S2 engine, tier off: 2.045 / 2.044 / 2.029, 2.043 / 2.045 / 2.033
Engine RTT p50 / p90: b 1.93-1.94 / 2.36-2.37 ms, c 0.65-0.67 / 0.78-0.84 ms; wait at the real lookup b 0.166-0.170, c
0.006-0.013 ms per layer; short requests b 40-71, c 82-228 of ~48k (the tail above).

Tracing (`serveidx`, off by default): `K3X_TRACE=1` (or a ring size) records per request, in a ring allocated at start:
idle wait, first byte, header and x received, x in host USM, each submission returned, each row in host memory and sent,
queue drained (CLOCK_MONOTONIC_RAW ns). `K3X_TRACE_PROF=1` adds the queue's profiling (each command's device start/end),
`K3X_TRACE_FREQ=1` the GT frequency before/after (a sysfs read on the request path, ~10 us). SIGUSR1 (while waiting) and
every connection close write TSV to `K3X_TRACE_OUT` (default `/tmp/k3x-trace.tsv`). The client's request number is
header word 16 when k < 16 (the engine's `LLAMA_EXPERT_STORE_REMOTE_TRACE`, `rrclient` with `RR_SEQ=1`).
`trace_report.py --server <dump> [--engine <dump>] [--client <RR_OUT>] [--conn N]` prints the per-stage breakdown and
joins the sides by request number (each clock compared only with itself). `rrclient` `RR_GAP_US` paces the requests
like the engine (one per layer, ~5-12 ms apart); back to back hides the idle costs (2026-10-28 M section).

Types: IQ1_S = 19, IQ2_XXS = 16. Results 2026-10-28 are in `docs/experiments.md`.

Image: `ghcr.io/xerootg/k3x-serve:g<sha>` (`docker/k3x-serve.Dockerfile`, workflow `k3x-serve.yml`) holds `k3x` built
by `build.sh` plus the pinned runtime above (the CI build of f3bcb80 is byte-identical to nuc2's `k3x-v8`). The cluster
runs it on nuc2 from argo-things `configs/kimi-k3/k3x-nuc2.yaml`.
