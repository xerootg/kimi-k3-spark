`g53cpu` computes GLM-5.3's unweighted routed expert rows. It supports raw GGUF
IQ2_XS, IQ3_XXS and IQ4_XS records with D=6144, F=2048; unsupported types fail
closed. `g53exact.h` / `g53ref_exact.cpp` are the unchanged scalar oracle.

Build and test from the checkout root, offline:

```sh
tools/glm53/cpu/build.sh
tools/glm53/cpu/test.sh "$HOME/glm53-work/data" "$HOME/glm53-work/mufu"
python3 tools/glm53/cpu/test_server.py
```

`test.sh` with no arguments runs synthetic tests and prints the extracted-data
skip. With DATA but absent MUFU it also tests the extracted zero/denormal cases.
The server suite automatically uses extracted slices and real tables when both
are available, otherwise explicitly reports the closed-form zero-gate fixture.
`G53_TEST_CLOSED=1` forces that CI fixture. No downloads or GPU are needed.
`G53_VNNI=' '` selects the exact AVX2 widening-dot fallback for older CI hosts;
the production default is a static executable with
`-march=x86-64-v3 -mavxvnni`, without AVX-512 (`G53_LINK=' '` selects dynamic linking).
All generated files are under this directory's ignored `build/` and log paths.

The integer grids/signs are decoded with AVX2; unsigned-by-signed VNNI dots
subtract an exact offset correction, including q8=-128. IQ2's two signed
truncating divisions and IQ3's signed truncating divisions precede float
conversion. Eight virtual CUDA threads are accumulated in each AVX vector.
The half-scale multiply, explicit FMA, warp additions and butterfly stay in
CUDA order; vector operations explicitly flush subnormals and canonicalize
NaNs. Quantization and MUFU SwiGLU use the scalar exact helpers, including the
float butterfly block sum stored as half. Tables are raw read-only mappings
(rcp/ex2p/ex2n only), prefaulted at startup; no approximate math fallback exists.

The persistent pool has preallocated q8/h scratch and 32-row tasks. It pins
workers when supplied CPUs and supports idle spinning then futex sleep and
optional SCHED_FIFO. Submit currently completes both phases synchronously;
wait/done therefore immediately succeed. It does not overlap socket writes
with compute. `lookup`/`interleave` are reserved and ignored; trace phase times
are unavailable. Records and output pointers must remain valid until run returns.

`g53cs IDX BIN PORT` accepts K3's 17-int header + 6144 floats and returns k rows
in request order. IDX is `gid gate_type down_type offset`; BIN can be a K3XS
v1/v2 store or raw records. Store metadata supplies the separate up type.
`K3X_CPU_CPUS=0-7` and `K3X_MUFU=<table dir>` are required configuration;
other RAM/disk/RT/trace knobs follow `tools/bitexact/cpu/k3cs.cpp`.
`K3X_DISK_TIER=1` enables the shared preallocated `expert_disk.h` cache;
`K3X_DISK_LRU_GB`, `K3X_DISK_THREADS`, `K3X_DISK_DEADLINE_MS`, and
`K3X_DISK_STORE` retain their K3 meanings. No disk tier means all-or-nothing
missing requests; with the tier enabled, valid rows survive partial misses.
Valid zero rows are returned unchanged. Missing rows are NaN sentinels.
`K3X_SETOUT` uses the store's expert count (256 for raw GLM records).
`K3X_FULLCHECK=1` checks resident records against the scalar oracle at startup.

The server warms all worker scratch/code/stacks before listening, prefaults the
disk cache, and requires successful `mlockall` before accepting work. Configure
an adequate memlock limit on the NUC (including tables, arenas and thread stacks).
`G53_ALLOW_UNLOCKED=1` is an explicit test/development escape; it cannot guarantee
residency. The allocation-instrumented server counts malloc/calloc/realloc,
alignment APIs and C++ new across request and compute threads. It also records
request page faults and rejects them when running with the lock requirement.
The inherited disk-cache unit suite checks pins, LRU, failed reads, concurrency,
deadlines and allocation gates. If local sockets are prohibited, the integration
suite reports the restriction and uses a compile-time pipe transport through
the same request loop; CI exercises real TCP.

Offline optimization benchmark (random grids, nonzero [-1,1] activations, zero gate;
no external data or MUFU tables):

```sh
tools/glm53/cpu/bench.sh 8 31
```

This links the original kernel from `G53_BASELINE_REF` (default HEAD) beside the
working kernel and alternates requests, reporting median before/after times.
Use `G53_BENCH_CPUS=0,1,3,6,12,13,14,15` and `G53_BENCH_CALLER=8` for the
165H's four P + four E configuration; adapt CPU IDs to the actual host.
The dynamic 32-row queue lets faster P workers claim more tasks. Pin one worker
per physical core, including E cores, before adding SMT or LP-E workers.
Prepared q8 bytes, exact integer bias sums and converted half scales are reused
across rows. IQ2/IQ4 integer scaling and the final float butterfly use SIMD; each float
operation retains the oracle's operands/order and FTZ/NaN handling. Completion
retains the original caller spin wait; idle workers use futexes. Startup warms
every worker and all row specializations. Benchmark/test fixtures use prefaulted
read-only file mappings and disable THP for that process only.
`G53_TEST_CLOSED=1 G53_TEST_PIPE=1 python3 tools/glm53/cpu/test_server.py`
forces closed-form fixtures and pipes for a fully offline server regression.
