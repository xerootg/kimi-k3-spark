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
