# Experiment ledger: Kimi K3 UD-IQ1_S on the DGX Spark and the GH200

Everything tried to make K3 faster, with the measured result and what we decided. The quality rule
for all of it: no change to which experts are used or to their weights (expert_used_count stays 16,
no requantization). Bit-identical output at temperature 0 is how we check that.

Engine details for each entry are in `patches/README.md`. Deployment numbers are in argo-things
`configs/kimi-k3/README.md` (Spark) and `configs/kimi-k3-gh200/README.md` (GH200).

Hardware:
- **Spark:** GB10, 128 GB unified LPDDR5X, one Gen5 NVMe at ~10 GB/s.
- **GH200:** 431 GiB LPDDR5X on NUMA node 0, H100 96 GB HBM3 as NUMA node 1, NVLink-C2C, virtio NVMe
  at ~9.5 GB/s.

## Current state (2026-10-19)

| | Spark | GH200 |
|---|---|---|
| image | g3ef5e51 | g231d482 |
| decode, coding prompt, warm | 1.60 tok/s (io_uring reads, self-prefetch, snapshot) | 13.6 tok/s (cache 424000 MiB, slot maps on the CPU) |
| cache | 52000 MiB host (hit rate ~60 %) | 400000 MiB host + 24 GiB HBM tier (hit rate ~96 %) |
| where a token's time goes | ~0.71 s: per layer, 4.2 ms NVMe reads + 3.5 ms GPU, serial | ~105 ms: expert matmuls over C2C, Q6_K spine, miss waits |

## Ledger

Verdicts: **kept** = in the deployed config. **opt-in** = in the engine behind a knob but off.
**dropped** = measured and rejected. **rejected** = not built, for the reason given.

### Both boxes / engine foundations

| what | result | verdict |
|---|---|---|
| Unpatched llama.cpp with the GGUF mmap'd (Spark) | 147 s/token: 4 KiB page faults, mmap read-around switched off by the kernel, direct reclaim | replaced |
| Patch 0001: MADV_WILLNEED of the selected experts | 4.45 s/token (33x better), I/O ~2.2 GB/s | superseded by 0003 |
| Patch 0002: GPU computes on the mmap'd experts in place (GB10) | 19 s/token: GPU reads of file pages go through UVM faults | opt-in, dead end |
| Pulsar (io_uring O_DIRECT expert streaming), Spark | 0.73-0.75 tok/s | speed reference only |
| RPC: experts of some layers computed on LAN nodes | 0.30 tok/s with 0001; with 0003, 0.68-0.70 vs 0.71 without RPC | dropped: LAN is 40-90x slower than the NVMe |
| Patch 0003 expert store: repacked records, host LRU bank, O_DIRECT pread pool | Spark 0.71 tok/s, hit rate 55 % | kept |
| Expert matmuls on the GPU straight from the registered host bank | Spark 1.43-1.47 tok/s (gap 12 -> 4 ms/layer) | kept |
| `--threads 8` vs 16 on the GB10 | 0.84 vs 0.71 tok/s (little cores off the critical path) | kept (8) |
| expert_used_count 12 instead of 16 (Spark) | 1.71 tok/s | rejected: changes the model's output |
| Speculative decoding while NVMe-bound | each draft token selects its own experts | rejected |
| vLLM / SGLang / ktransformers | cannot run K3 on these boxes (ktransformers is x86 AMX only) | rejected |

### GH200 bring-up and the first speed pass (2026-10-18/19)

| what | result | verdict |
|---|---|---|
| 470 GB cache | global OOM: "525 GiB" is 431 GiB host + the 96 GB HBM | fixed: 360000, later 400000 MiB |
| 32-bit expert offsets in MMQ/MMVQ | illegal memory access past slot ~13.7k | fixed (64-bit pointer math) |
| Records read as 1 MiB pieces across 32 threads (`LLAMA_EXPERT_STORE_CHUNK_KIB`) | one 6.6 MB read 2.9 ms -> 1.2 ms; decode 7.0 -> 8.5-8.9 tok/s | kept |
| Slot maps on the GPU + CUDA host-bank hook (186 graph splits -> 2) | small gain: the splits were not the cost | kept |
| MMQ walks only the experts a batch uses | prefill 2048 tok 15.5 -> 64 tok/s (with ubatch 2048) | kept |
| Scan-resistant LRU for prefill | protects the decode working set; first version caused an Xid 31 MMU fault with 2 clients | kept, fixed |
| Page cache of the mapping released after load | ~45 GB more for the cache | kept |
| Cache regions `mbind`-ed to the CPU node | a 390 GiB populate had spilled into the HBM and the compute buffers failed to allocate | kept |
| ubatch 2048 with prefill from the bank (`LLAMA_EXPERT_STORE_MAX_BATCH`) | prefill 512 / 1024 / 2048 tok: 54 / 90 / 126 tok/s (was 23 / 20 / 16) | kept |
| n-gram speculation (`--spec-type ngram-mod`) | coding 7.9 -> 8.05 tok/s, not bit-identical | dropped |
| Asynchronous host-bank hook (GPU waits on a mapped mailbox, CUDA graphs back on for 97 % of graphs) | same speed: launch overhead was not the bottleneck | kept as the base, neutral |
| Predictive prefetch of the next layer's experts (`LLAMA_EXPERT_STORE_PREFETCH=1`) | 83 % of predicted reads used, misses -20 %, decode 7.53 vs 7.75 | opt-in: the GH200's ~1 ms GPU gap is shorter than a read |
| Lazy 6.7 GB prefill staging everywhere | the Spark thrashed for 40 min on its first long prompt | eager where prompts use it |

### GH200 second pass (2026-10-19)

| what | result | verdict |
|---|---|---|
| Op profile, warm decode (`GGML_CUDA_OP_PROFILE`) | per token: 26 ms expert matmuls over C2C, 24 ms miss waits, 26 ms Q6_K spine, ~30 ms small ops | guide for the rest |
| Q6_K GEMV on the H100 (test-backend-ops) | 2.2 TB/s (f16 3.45 TB/s): the spine runs at ~60 % of HBM bandwidth | open |
| Hopper MMVQ: the GB10 L2 prefetch and/or the GB10 parameter table for sm_90 | Q6_K 21.6 us -> 22.2 / 22.3 / 23.7 us | dropped |
| UVM / ATS / managed memory / THP for host reads (docs/uvm-experiment.md) | fault-free ATS = registered (345 GB/s); every fault path is slow; host pages on the HBM node read at 2.1-2.9 TB/s | led to the HBM tier |
| HBM tier, 8 / 16 / 24 GiB (`LLAMA_EXPERT_STORE_HBM_MIB`) | 8.39 / 8.87 / 9.31 tok/s vs 7.77 (21 / 30 / 37 % of lookups from HBM) | kept |
| THP on the HBM pages | first reads 2.1 -> 2.6 TB/s; with the faster swaps 9.31 -> 9.49 tok/s | kept |
| Swaps in 256 KiB pieces (all I/O threads) instead of 1 MiB | 0.26 -> 0.14 ms per swap | kept |
| Use-count decay every 32 / 128 / 512 tokens | 9.19 / 9.49 / 9.58 tok/s (112k / 38k / 15k swaps) | kept: 512 |
| Swaps per layer fill: 1 vs 2 | 9.49 vs 9.49 | 2 (default) |
| HBM tier above 24 GiB | ~6.5 GB of HBM left after load; the compute buffers are allocated after the cache | not tried, risk of a failed load |

### Spark read path (2026-10-19)

The Spark perf-hunt workflow ranked 10 lossless candidates (prefetch, poller sleep, shared expert under the I/O
wait, CPU placement, earlier prediction, preemptible prefetch, NVMe sweep, futex handoffs, cache headroom). It was
stopped during the first experiment to work on the reads directly; the rest are still open.

| what | result | verdict |
|---|---|---|
| Predictive prefetch on the Spark (`LLAMA_EXPERT_STORE_PREFETCH=1`, plus reordered and capped variants) | 64 % of the predicted reads used, hit rate 59.5 -> 74 %, but decode 1.33-1.35 vs 1.39-1.43 tok/s: the wasted reads compete with the real ones on a saturated drive | dropped |
| NVMe ceiling, fio random reads (Samsung PM9E1, Gen5 x4) | over the store's real blocks: 10.5-11.0 GB/s by any path (ext4, raw device, NVMe passthrough); 14.9 GB/s only over never-written blocks, which the drive answers without reading flash | ceiling |
| Read shape microbenchmark (`tools/readbench.cpp`, 7-record fills) | pread pool 1 MiB: 4.52 ms; pool 4 MiB / whole records: 4.8-4.9 ms; io_uring any piece size, qd >= 32: 3.86-3.95 ms; fixed buffers = plain | io_uring kept |
| Hugepage vs 4 KiB destination pages | 3.9 vs 4.5 ms per fill (one IOMMU mapping per page) | see THP row |
| io_uring reads in the engine | Spark decode 1.37 -> 1.47 tok/s, reads 9.6 -> 10.5-11.1 GB/s, bit-identical | kept |
| I/O pool reuse race (previous run's worker grabbing a job of the next) | fixed: run() waits for the workers to leave | fixed |
| THP on the host cache regions (on since the HBM pass) | Spark: 4 of 7 starts gave NaN from the first prefill ("GGGG..."), 0 of 4 without | off by default (`LLAMA_EXPERT_STORE_THP=1`) |
| mlock of the bank to keep its huge pages (reclaim split ~75 % of them during load) | not measured; without THP it buys nothing | dropped |
| Lossless compression of the records (decompress on the GPU) | zstd -3/-19: 0 %; IQ1_S split into fields: 1.6 %; order-0 entropy bound: 3.4 % (the grid indices are ~8.0 bits/byte) | rejected |
| Self-prefetch (`LLAMA_EXPERT_STORE_SELF_PREFETCH=1`): post the layer's real expert ids right after the router, run the shared expert and the latent down projection, then the real lookup | ON 1.605 / 1.606 / 1.598 and 1.599 / 1.360 (one stall) / 1.610 tok/s; OFF 1.490 / 1.495 / 1.491 and 1.485 / 1.486 / 1.489 (+7.7 % on the medians). The poller reads 6.7 records in 4.0 ms per post while the GPU computes; the real fill drops to 0.12 ms (0.2 records), the layer from ~7.7 to ~7.0 ms. md5 gold on every run, same build both legs | kept, off by default (flag) |

### GH200 third pass (2026-10-20)

All runs: deployed HBM config (24 GiB, decay 512), 3 x 800-token coding generations, md5 ac78af0e5d every time.

| what | result | verdict |
|---|---|---|
| New engine (io_uring, whole records) vs deployed g231d482 | 7.23 vs 9.59 tok/s, reads per layer 1.01 vs 0.58 ms | io_uring off on the GH200 |
| io_uring with 1 MiB / 128 KiB pieces | 8.36 / 9.41 tok/s | dropped: the virtio disk does better with 32 submitting threads (one blk-mq queue per CPU) |
| New engine, thread pool (`LLAMA_EXPERT_STORE_URING=0`, race fix) | 9.53 / 9.52 / 9.65 tok/s | same speed, has the race fix |
| THP on the host regions (`LLAMA_EXPERT_STORE_THP=1`) | 7.26 vs 7.23 tok/s | no effect on the 64 KiB-page kernel |
| Pieces of 256 / 512 KiB instead of 1 MiB (pool) | 9.67 / 9.59 tok/s, reads per layer unchanged | no effect |
| Read microbenchmark on the virtio disk | 0.35-0.47 ms per 6.6 MB record once warm, any piece size, either NUMA node as destination; the first test of each run is 0.7-1.3 ms (warm-up), which first looked like a 4x win for small pieces | no lever |
| `GGML_CUDA_GRAPH_OPT=1` | 9.52 tok/s | dropped |
| Cache policy, simulated on a real decode trace (`LLAMA_EXPERT_STORE_TRACE`, scratch cachesim.py), decode after warm-up | LRU 98.21 % hits, ARC 98.27 %, LFU with decay 98.36 %, Belady (needs the future) 99.58 % | rejected: a better policy cuts misses ~8 %, under 2 % of tok/s |
| Op profile, warm decode (profiler on) | Q6_K spine ~26 ms, IQ1_S expert matmuls ~82 us per call for ~34 MB (close to C2C + HBM speed, ~3 ms to gain at most), misses the largest item | guide |
| Q6_K GEMV for Hopper: TMA-staged weight tiles, stock math (tools/q6k-gemv-bench.cu) | isolated: 2.45 -> 2.88 TB/s (output head 3.39), bitwise identical on 45 shape x seed cases; in the server unfused Q6_K 20.8 -> 19.2 ms/token, fused gate/up 5.4 -> 5.7 ms; end to end within noise (9.48-9.74 tok/s) | kept for unfused GEMVs |
| The Hopper Q6_K TMA GEMV on the GB10 (`GGML_CUDA_MMVQ_Q6K_TMA=2`, shared memory limit from the device) | test-backend-ops 4096x14336: 210-221 us stock vs 197-210 us (5-10 %, ~1.5-2 % of a Spark token by estimate); NOT bit-identical there (24 of 45 bitcheck outputs differ, same error vs CPU): the GB10's stock MMVQ uses its own warp layout, the TMA kernel mirrors Hopper's | opt-in only; a GB10-layout variant would be needed |
| Q6_K variants that failed (all bit-identical): deeper register pipelining, L1-bypass / evict-first hints on stock loads, 2-4 rows per block, 3-8 copy stages, 8 blocks per SM, split-row warps, mbarrier instead of syncthreads, fewer instructions | 25-48 us vs 25.1 us for the kept variant (stock 29.5) | dropped |
| PDL coverage of the decode kernels | already on for nearly all hot ops; argsort and softmax (~280 launches/token) still raw | not worth it |
| Simulated LRU miss rate vs cache size (decode trace) | x1.00 1.79 %, x1.01 1.65 %, x1.02 1.51 %, x1.03 1.35 %, x1.04 0.68 %, x1.08 0.58 %, x1.15 0.41 %: this workload's working set sits just above the deployed cache | led to the next rows |
| Cache 412000 / 416000 MiB with the prompt cache on (8 GiB default) | 10.45 tok/s at 412000, then host memory ran out on the third run (watchdog kill at 1.5 GB available) | prompt cache off |
| Cache 412000 / 416000 MiB, `--cache-ram 0` | 412000: 10.45 / 10.50 tok/s, hit 96.9 %; 416000: 11.53 / 12.31 tok/s, hit 97.3 %; lowest MemAvailable 2.7 GB; bit-identical | kept: 416000 (repeated-prompt benchmark, varied traffic gains less) |

### Cache snapshot (2026-10-21)

| what | result | verdict |
|---|---|---|
| Write the cache bytes to the SSD at shutdown and read them back | rejected before building: the bytes are already in the store on the same SSD, re-reading from the store costs the same and skips a 400 GB write | rejected |
| Snapshot of which experts are resident (recency + use counts, ~6 B/slot), re-read from the store at start (`LLAMA_EXPERT_STORE_SNAPSHOT`) | Spark: all 7980 slots restored, 52.5 GB in 5.5 s (9.6 GB/s); hit rate 84.7 % over the first 84k lookups vs 47.7 % cold; first run 1.59 tok/s; md5 unchanged, both hooks. GH200: 67647 slots, 445 GB in 47 s; early hit rate 90 % vs 60 % cold; first coding run 9.0 -> 9.2 tok/s only | kept on the Spark |
| GH200: snapshot restore together with the HBM tier | with GPU slot maps (async hook) the reference md5; with the slot maps on the CPU, or the sync hook, the restored cache gives a different answer (10860c30c1), while a cold start in those modes gives the reference | held back on the GH200 until understood: something in the restored HBM slots (bytes vs keys, or GPU cache lines of the HBM-node pages) is only coherent through the async path's memop wait |

### GH200: two sequences decoding together crash with CUDA graphs on (found 2026-10-21)

| what | result | verdict |
|---|---|---|
| `gh200-native-test.sh "1"` (one 1-token completion, one chat, then two concurrent 64-token chats) | CUDA "illegal memory access" in the first 2-token decode graph, every time, with the current build, with the self-prefetch graph changes reverted, and with the g231d482 binary from 2026-10-19 (`"1 64"`, a 64-token prompt first, passes: shape-order dependent) | long-standing; production has `--parallel 2`, so two clients at once could crash the pod |
| Same with `GGML_CUDA_DISABLE_GRAPHS=1` | still crashes; it had only passed together with `GGML_CUDA_DEBUG_SYNC=1` (every node synchronized), so it is a race, not a CUDA graph bug | |
| Isolation: synchronous host-bank hook (`GGML_CUDA_HOST_BANK_ASYNC=0`) / no HBM tier / graphs off + no HBM tier / `--parallel 1` / slot maps on the CPU (`LLAMA_EXPERT_STORE_GPU_SLOTS=0`) | passes / crashes / crashes / passes / passes | the race is in the asynchronous mailbox path with two sequences in a batch; root cause open |
| Output check of the sync hook (md5 of the 800-token coding run, reference ac78af0e5d) | sync hook + HBM tier: **10860c30c1, a different answer**; sync hook without the tier, async hook with or without the tier, slot maps on the CPU with the tier: all ac78af0e5d; the Spark gives its reference with both hooks | the sync hook is NOT a valid mitigation here: the HBM tier's swaps are unsafe in it (the GPU idles with nothing queued while the CPU rewrites a slot; the async path's memop wait orders it). Candidate mitigation: slot maps on the CPU, which also measured faster (12.5 vs 11.5 tok/s in one run) |

### GH200 miss waits (2026-10-21, snapshot restored before every leg, 3 coding runs, runs 2-3 = steady state)

| what | result | verdict |
|---|---|---|
| Spin-wait in the I/O pool (`LLAMA_EXPERT_STORE_SPIN_US=2000`) | 11.54 vs 11.47 tok/s | neutral, opt-in; revisit |
| 512 KiB read pieces instead of 1 MiB | 11.65 tok/s | noise |
| Cache 424000 MiB with `vm.min_free_kbytes` 11 GB -> 2 GB on the node (the 64 KiB-page kernel's default reserve is 11 GB) | 13.45 tok/s, hit rate 98.2 -> 98.7 %, reads per layer 0.34 -> 0.23 ms, lowest MemAvailable 8 GB, no watchdog kill | kept (pending deploy) |

### Revisit when the bottleneck moves

Neutral is not dead: these are lossless and stay in the engine behind a flag, but measured neutral only because
something else dominated at the time. Re-test them after any change that shifts the per-token breakdown.

| change | why it was neutral | re-test when |
|---|---|---|
| Predictive prefetch (`LLAMA_EXPERT_STORE_PREFETCH=1`) | GH200: the ~1 ms GPU gap was shorter than a read; Spark: the NVMe was saturated, wasted reads competed | GH200 misses are rarer now (cache 416000); Spark reads got faster |
| Hopper Q6_K TMA GEMV (on by default) | -1.6 ms/token in the op profile, but end to end within noise while miss waits dominated | after miss waits shrink further |
| `GGML_CUDA_GRAPH_OPT=1` | parallel attention branches hidden behind the expert waits | GPU-bound phases (fewer misses, prefill) |
| Async host-bank hook / CUDA graphs | launch overhead was not the bottleneck | when GPU time per layer dominates |
| Eviction policy (ARC / LFU decay, simulated) | ~8 % fewer misses at the GH200's capacity | tighter caches (the Spark's 10 %) |
| Hugepage destinations for O_DIRECT (3.9 vs 4.5 ms per fill) | THP on host regions gave NaN on the GB10 | once that NaN is understood |
| io_uring on the GH200 | one submitter = one virtio queue | with multiple rings across CPUs |

### Open ideas (not yet measured)

- **The rest of the Spark perf-hunt list.** Shared expert and routed_down under the I/O wait (self-prefetch with
  the real ids, est. +6 %), poller falling asleep inside the GPU gap (+2 %), I/O and poller threads on the X925
  cores (+2 %). The reads now run at the drive's ceiling, so more overlap is the remaining lever on the read side.
- **Why THP on the host regions gives NaN on the GB10.** Fixing it would bring back hugepage destinations
  (3.9 vs 4.5 ms per fill), but only if the pages also survive the load's reclaim.
- **Frequency-aware eviction for the host cache.** The routing skew that makes the HBM tier work should also
  help a policy like LFU or ARC beat plain LRU, which matters most on the Spark's 10 % cache.
- **Q6_K GEMV for Hopper.** The spine reads at 2.2 of ~3.5 TB/s, so a better kernel could save ~10 ms/token.
- **15+ tok/s on the GH200.** Only path found so far: all experts resident in memory, for example a second
  GH200 with expert parallelism.
