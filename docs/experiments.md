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

## Current state (2026-10-21)

| | Spark | GH200 |
|---|---|---|
| image | g3ef5e51 | gb4c74d2 |
| decode, coding prompt, warm | 1.80 tok/s (io_uring reads, self-prefetch, speculative prefetch v2, snapshot) | 13.65-13.84 tok/s (cache 424000 MiB, slot maps on the CPU, snapshot restored at start; run 1 13.7 tok/s with `cache_prompt: false`) |
| cache | 52000 MiB host (hit rate ~60 %) | 424000 MiB host + 24 GiB HBM tier (hit rate ~99 % warm) |
| where a token's time goes | ~0.55 s: per layer, 3.0 ms of reads the GPU waits on + 2.8 ms of GPU work with ~4.8 predicted records read under it | ~105 ms: expert matmuls over C2C, Q6_K spine, miss waits |

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

### Spark: speculative prefetch v2, reads under the GPU's part of the layer (2026-10-21)

The question was whether the ~3 ms of GPU work per layer that runs with the drive idle could carry reads.
An nsys profile of 24 decode tokens (native build, production args, self-prefetch on, `--cuda-graph-trace=node`,
2207 layers measured, medians) put the layer at:

| segment | median | what runs |
|---|---|---|
| self-prefetch post -> real lookup post | 0.65 ms | shared expert, latent down projection; the poller reads the misses |
| real lookup post -> next kernel | 3.42 ms (mean 3.91, p90 6.05) | the GPU waits for the fill (4.1 ms of reads minus the 0.65 above) |
| after the lookup -> next self-prefetch post | 2.82 ms, kernels busy 2.88 | expert matmuls, routed_up, next layer's attention and router: **the drive is idle** |
| layer | 6.67 ms (mean 7.40) | |

Kernels are busy 47 % of the token. The drive-idle window is 2.8 ms = ~4.8 records at the 11 GB/s ceiling, against
6.7 misses per layer; filling it with the right records would take the self-prefetch fill from 4.1 to ~1.5 ms.

The first predictive prefetch had the prediction, not the overlap: `prefetch_decode` ran the speculative reads through
the I/O pool on the poller thread at full depth, so the next real post queued behind them and they saturated the drive
exactly when the real reads arrived. Version 2 (patches/README.md, "Predictive prefetch"): a side reader thread with
its own io_uring ring and a bounded number of pieces in flight, posted right behind the real lookup, preempted by every
real fill (started records are finished and adopted, unstarted ones handed back), best router scores first, LRU
touched only on the poller thread, per-slot states keep its slots out of the victim and HBM-exchange paths.

Two bugs on the way, both found by the gold md5 and the counters: the reader took an SQE before knowing whether it had
work, and an SQE taken and not prepared goes out as a no-op whose completion (res 0, user data 0) was booked as "piece 0
read 0 bytes" (looked like a kernel quirk until the counters showed every prediction handed back unread: the in-flight
count had underflowed and the reader was waiting forever). Zero-length completions are now retried, bounded, with a
warning; none seen since.

All legs below: `~/spark-bench.sh`, 3 x 200-token coding runs after a 32-token warm-up, self-prefetch on, md5
`e801484d` on every run and `b2c79009` on every warm-up (gold). Prompt speeds were noisy this day (0.4-1.3 tok/s for the
same prompts that did 2.6 before, in the baseline legs too), decode was not.

| leg | decode tok/s (runs 1/2/3) | predictions: queued per post, started, used | self-prefetch fill |
|---|---|---|---|
| base (self-prefetch only) | 1.577 / 1.525 / 1.606 | | 6.7 records, 4.02 ms |
| v2, depth 8, router input (`PREFETCH=1`) | 1.689 / 1.755 / 1.756 | 6.16, 4.5, 49 % of queued (67 % of started) | 4.5 records, 3.14 ms |
| v2, depth 8, FFN input + shared expert under the next norm (`PREFETCH_INPUT=1`) | 1.771 / 1.812 / 1.791 | 7.00, 4.76, 47.5 % of queued (70 % of started) | 4.1 records, 3.04 ms |
| v2, depth 4 | 1.727 / 1.658 / 1.707 | 6.29, 3.6, 2.3 used per post | 4.8 records, 3.27 ms |
| v2, depth 16 | 1.545 / 1.376 / 1.742 (a stall in run 2, prompt at 0.1 tok/s) | 6.10, 5.0, 3.3 used per post | 4.3 records, 3.26 ms |
| v2, depth 8, unused predictions demoted (`PREFETCH_DEMOTE=1`) | 1.752 / 1.765 / 1.758 | 7.41, 5.1, 2.4 used (32.6 % of queued: demoted predictions are evicted and predicted again) | 4.4 records, 3.15 ms |
| base again | 1.525 / 1.582 / 1.599 | | 6.7 records, 4.02 ms |
| v2, depth 8, input 1, demoted | 1.750 / 1.804 / 1.802 | 7.66, 5.1, 2.8 used (36 %) | 4.0 records, 2.99 ms |
| v2, depth 16, input 1 | 1.592 / 1.745 / 1.767 | 6.94, 5.4, 3.7 used (53 %) | 3.9 records, 3.07 ms |
| base, third time | 1.235 / 1.422 / 1.611 (stalls in runs 1-2, prompt at 0.1 tok/s) | | 6.7 records, 4.17 ms |
| v2, depth 4, input 1 | 1.756 / 1.444 / 1.764 (stall in run 2) | 7.12, 3.6, 2.6 used | 4.6 records, 3.16 ms |

**Verdict: kept, deployed as `LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=1` (depth 8, no
demote): 1.60 -> 1.80 tok/s on the steady-state runs (+12 %), bit-identical.** Depth 8 already runs the drive at its
ceiling inside the 2.8 ms window (~4.8 records started per post); depth 4 starts fewer, depth 16 starts a few more but
finishes more of them past the preemption, which the self-prefetch fill then waits on (0.56 ms per adopted wait), net
nothing. Demoting unused predictions costs precision (they get evicted and predicted again) and gains nothing at this
cache size. The FFN-input predictor is worth 3 % over the router-input one. What is left: the window is full and
~30 % of the started reads are wasted; a more precise predictor (the real next input is this plus the routed output and
the next attention) is the only lever left on this path, worth up to ~1.9 instead of ~4 records in the self-prefetch
fill (layer ~4.0 instead of ~5.9 ms). The box stalled a few times this day (a 200-token run at 1.2-1.4 tok/s with the
prompt at 0.1 tok/s, in baseline legs as well); those runs are listed, not used.

### Spark: what the GPU does in the drive-idle window (2026-10-21, from the same nsys capture)

The 2.82 ms per layer between the real lookup and the next layer's self-prefetch post, broken down by kernel over
2207 layers (`nsys-kernels.py` on `k3prof.sqlite`): kernels 2.85 ms, gaps between kernels 0.07 ms (median 0.4 us,
p90 8.7 us). Launch latency, the host hook and the poller handoff are not in it; it is weight streaming:

| kernel | ms per layer | calls per layer | what |
|---|---|---|---|
| `mul_mat_vec_q` | 2.39 | 11 | expert and dense GEMVs, see below |
| `mul_mat_vec_f` | 0.12 | 1.7 | the router (896 x 7168, f16) and small f32 ones |
| `mul_mat_vec_q_moe` | 0.10 | 0.1 | the dense FFN of layer 0 (33792 x 7168 x 3, once per token) |
| everything else (norms, quantize, bcast, argsort, KDA `gated_delta_net`, `ssm_conv`, flash attention) | 0.25 | ~50 | |

One KDA layer's GEMVs in order: the routed experts 768x16 / 768x16 / 896x16 blocks at 188 / 208 / 210 us (16 experts
x 7.8 MB = 125 MB from the host bank in 0.6 ms = **208 GB/s**, i.e. the 4 KiB-page bank is read at the same
efficiency as the GPU's own buffers: no TLB cost, the hugepage hypothesis for the GPU side is dead), routed_up 113 us,
then the next layer's attention: three 12288-row projections at ~345 us each, one more at 380 and a 7168-row one at
335 us, together ~1.75 ms for ~370 MB of Q6_K at ~210 GB/s. An MLA layer's attention is ~1.1 ms (117 / 50 / 116 /
41 / 18 / 31 / 396 / 343 us). The output head (163840 rows, 4.7 ms) and layer 0's dense FFN (3 x 0.93 ms) run once
per token. The GB10's LPDDR5X peaks at ~273 GB/s, so the dense streaming runs at ~77 % of it.

Conclusions: the GPU side of the layer has no structural slack; the next layer's projections cannot start before this
layer's experts have landed, and they are the window the speculative prefetch now fills. Kernel-level gains are
bounded by the ~23 % between 210 GB/s and the memory peak, and the one candidate measured, the Hopper TMA Q6_K
kernel on the GB10, gives 5-10 % of that per kernel (~1.5-2 % of a token) and is not bit-identical there (see the
GH200 second pass). The THP-on-the-bank question is therefore only about the O_DIRECT fill (3.9 vs 4.5 ms per
7-record fill, ~0.4 ms of a ~5.9 ms layer at the current 4.1-record fill), not about the GPU. The IOMMU of the
NVMe is in `DMA-FQ` mode (`iommu.passthrough=0` on the kernel command line); `iommu.passthrough=1` would remove
the per-page map cost of every fill without any hugepage, at the price of a node reboot (not tried, the user's
call).

### Spark stalls: swap and the prompt cache (2026-10-21)

The question was the "0.1 tok/s" stretches seen during the prefetch v2 A/B (a 4-token prompt taking 20-40 s, a
200-token run at 1.2-1.4 instead of 1.8 tok/s, in baseline legs too). Setup: `spark-bench-stall.sh` (streaming
completions with per-token timestamps, gaps over 1.5 s logged) plus a 1 s sampler of `/proc/vmstat`, PSI, swap,
the server's `VmSwap`, the NVMe temperature and the D-state threads with their `wchan`. Six 300-token runs per leg,
the deployed engine config, 300-token output prefix-identical to the gold 200-token text on every run.

What the box looks like after a load: 114 of 121 GB used (bank 52 GB + staging 6.7 GB + the spine and buffers on
the GPU ~45 GB), MemAvailable 7-8 GB, `vm.swappiness` 60, 16 GB swap file on the model NVMe, NVMe at 39-42 C with
no throttle events (not thermal). The load itself (spine streamed through the page cache, bank populated) swaps out
cold pages: the server had **152 MB in swap** after every load, an 86 MB anonymous region almost whole, 44 MB of the
heap, parts of libcublasLt's code, and the lifetime counters show it (93 M swap-ins, 76 M swap-outs, 34 M allocation
stalls, 11 G pages scanned by direct reclaim since boot). A swapped page comes back as a major fault served behind
the expert reads that keep the NVMe queue full, ~10 ms each; a few hundred in a row are a multi-second stall.

| leg | result |
|---|---|
| A: as deployed, 6 x 300 tokens | 1.82 tok/s every run, no gap over 1.5 s in any run; the warm-up shows a 3.8 s "gap" at its token 25 |
| B: + `LLAMA_EXPERT_STORE_MLOCK=1` (new, default on): the engine mlocks its memory except the model file mappings at the first fill | locked 63.6 GB in 5-17 s, `VmSwap` 0 from then on; 1.81-1.82 tok/s, no gaps; the same 3.7 s warm-up gap |
| C: 8 different prompts x 100 tokens, prompt cache at its default 8 GiB | cache grows 1335 MiB per saved prompt to 8008 MiB; **MemAvailable down to 1.4 GB, 5.4 GB swapped out** during the eight requests; no stall here because no prompt was reloaded |
| D: same prompts, `--cache-ram 0` | MemAvailable never below 6.4 GB, 1 GB swapped out (all during the load); decode and prompt speeds the same as C (1.80-1.92 tok/s, ttft 12.8-22.7 vs 14.1-20.7 s) |

Findings:
- The warm-up "gap" is not a stall: the server's own timing shows 32 tokens at a steady 526 ms, and the client got
  28 pieces. It is the chat parser holding the tokens around the end of the reasoning block until the tag is
  complete. The old benches could not see it (non-streaming).
- The stall of the earlier day did not reproduce in 28 runs over four legs, but its mechanism did: the prompt cache.
  Every saved slot of this model costs ~1.3 GiB of host RAM (KV plus the 443 MiB recurrent-state checkpoints), the
  default limit is 8 GiB, and the box has 7 GB to give, so a filling cache swaps out whatever is cold, including the
  cache's own entries, and a conversation that comes back loads its entry one 4 KiB fault at a time behind the
  expert reads. The earlier A/B restarted the server a dozen times (each load swapping more out, 559 MB of swap in
  use by the end) and re-sent the same prompt, which is exactly the reload path.
- The memory lock costs nothing at steady state (same tok/s) and takes the server's own pages out of that game
  for good. The prompt cache is turned off on the Spark as it already is on the GH200 (`--cache-ram 0`): with two
  slots the slots themselves keep their context, the cache only helps a third conversation.

**Verdict: deployed, `--cache-ram 0` and the lock (the pod gets `IPC_LOCK`), bit-identical.** Not a tok/s change;
a removal of the swap path. Still open: a reproduction under production traffic (several conversations) would show
whether anything else stalls; the sampler and the streaming bench are on the box for that.

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

### GH200: the two-sequence crash root-caused and fixed; the "wrong md5" was the prompt cache (2026-10-22)

**The crash.** The asynchronous host-bank hook passed the *lookup node* (a `ggml_tensor *` of the scheduler's
graph) through the mailbox, and the poller dereferenced it when the GPU reached the request. With eager launches
(no CUDA graph: any batch over one token) the host is far ahead of the GPU; when a decode has more than one ubatch
it has already rebuilt the next ubatch's graph in the same scheduler memory, so the poller read a tensor of the
next graph, filled the wrong layer or none, and the lookup read -1 as an expert slot. Two sequences make a
multi-ubatch decode whenever a second client's prompt joins a running sequence (the server batches the first
sequence's decode token with the second's prompt and `split_equal` splits that by sequence), which is why the crash
needed `--parallel 2`, two clients, and a particular request order. Fix: the mailbox carries the slot map (a tensor
the cache owns) and the kind of lookup by value, never the node (`ggml_backend_cuda_host_bank_hook_t` changed).
Diagnostics added on the way: `GGML_CUDA_HOST_BANK_CHECK=1` validates every lookup's slot indices on the GPU.

| what | result |
|---|---|
| Reproduction, old mailbox (engine db2abf1 + the check code, async hook, GPU slot maps, HBM tier, cache 416000): 1-token completion, 64-token chat, two concurrent 64-token chats | crashed 2 of 2 ("illegal memory access"); the log shows the second client's prompt joining slot 1 at a different `n_past` than slot 0 |
| Same with `GGML_CUDA_HOST_BANK_CHECK=1` (synchronizes at the end of every graph) | passed: the end-of-graph sync keeps the host from rebuilding the graph under the GPU, exactly the race |
| compute-sanitizer / GPU core dump (`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`) | not usable: the sanitizer serializes kernels (hides the race); the dump of the 96 GB device grew at 1.6 MB/s and cuda-gdb 12.8 cannot read the 580 driver's format |
| With the fix (slot map + kind through the mailbox), same sequence | passed 7 of 7 (every leg below, plus `gh200-native-test.sh "1"` at cache 424000 in async, snapshot, CPU-slot and sync modes) |

**The md5.** Every leg of the fixed build, in every mode (async + GPU slot maps, slot maps on the CPU, sync hook,
with and without the HBM tier, with and without the snapshot, cache 416000 and 424000, TMA off), gave 10860c30c1
for the 800-token coding run instead of the reference ac78af0e5d, while the old engine (c9b8f74-era binary) gave
ac78af0e5d in a shorter sequence. Both answers are legitimate: they differ at token ~80 and depend on **what the
server's prompt cache holds**. The coding prompt restores a 67-token context checkpoint (the chat template's prefix)
computed by an earlier request, and a prefix whose KV came from the two-client prefill (a two-sequence ubatch) is
not bit-identical to one computed in a single-sequence prefill; the rest of the prompt is then processed on top of
it. Which checkpoint the coding prompt lands on depends on the slot the server picks (LRU, timing-dependent with
two concurrent clients), which is why earlier legs of the *same* configuration gave either value and why the
"sync hook + HBM tier" and "snapshot restore" rows above looked like wrong answers.

| leg (fixed build unless noted, cache 424000, full sequence incl. two clients before the coding runs) | md5 |
|---|---|
| async hook, GPU slot maps, HBM tier, cold / snapshot restored | 10860c30c1 / 10860c30c1 |
| slot maps on the CPU (deployed mode), cold | 10860c30c1 (13.5 tok/s) |
| sync hook, HBM tier; no HBM tier; `GGML_CUDA_MMVQ_Q6K_TMA=0`; cache 416000 | 10860c30c1 every time |
| **old engine** (c9b8f74 era), slot maps on the CPU, same full sequence | **10860c30c1** |
| old engine, async, short sequence (1-token completion, 8-token chat, no two clients) | ac78af0e5d |
| engine db2abf1 (without the mailbox fix), slot maps on the CPU, short sequence | ac78af0e5d |
| fixed build, async hook, GPU slot maps, HBM tier, full sequence, **`cache_prompt: false`** on the coding runs | **ac78af0e5d** (9.5 / 12.7 tok/s) |
| fixed build, slot maps on the CPU, short sequence | ac78af0e5d |
| fixed build, async hook, GPU slot maps, HBM tier, **snapshot restored** (68875 experts, 453 GB in 47.5 s), full sequence, `cache_prompt: false` | ac78af0e5d x 3 (10.3 / 12.8 / 12.8 tok/s) |
| fixed build, **slot maps on the CPU + snapshot restored**, full sequence, `cache_prompt: false` | ac78af0e5d x 3 (**10.8 / 13.65 / 13.67 tok/s**, hit rate 99.0 %) |

Deployed from this: slot maps on the CPU (still the faster mode on the GH200, 13.65 vs 12.8 tok/s steady) plus the
snapshot (run 1 at 10.8 instead of ~9.9 tok/s, and the cache warm after a restart). The asynchronous path is safe
again and stays the base for the Spark's prefetch.

Protocol from here: GH200 reference runs use `cache_prompt: false` (or a fresh server), the gold stays
ac78af0e5d. The Spark gold runs (`spark-bench.sh`) send the same fixed sequence every time (one warm-up, then the
runs), so their md5s are reproducible by construction; a different request history there would move them the same
way. The "sync hook wrong answer" and "snapshot held back" verdicts above are withdrawn.

### GH200 miss waits (2026-10-21, snapshot restored before every leg, 3 coding runs, runs 2-3 = steady state)

| what | result | verdict |
|---|---|---|
| Spin-wait in the I/O pool (`LLAMA_EXPERT_STORE_SPIN_US=2000`) | 11.54 vs 11.47 tok/s | neutral, opt-in; revisit |
| 512 KiB read pieces instead of 1 MiB | 11.65 tok/s | noise |
| Cache 424000 MiB with `vm.min_free_kbytes` 11 GB -> 2 GB on the node (the 64 KiB-page kernel's default reserve is 11 GB) | 13.45 tok/s, hit rate 98.2 -> 98.7 %, reads per layer 0.34 -> 0.23 ms, lowest MemAvailable 8 GB, no watchdog kill | kept (pending deploy) |

### GH200: speculative prefetch v2 under the CPU split (2026-10-22)

The Spark's prefetch v2 needed two things the GH200 config does not have: the slot maps on the GPU (the prediction
was posted through the host-bank hook) and io_uring (one ring, which on the virtio disk is one blk-mq queue). Both
are now in the engine: with `LLAMA_EXPERT_STORE_GPU_SLOTS=0` the prediction is computed on the device before the
layer's real router and its lookup sits behind the real lookup in the same CPU graph (no extra split: 186 per
graph, as before), where the CPU graph hook hands it to the side reader right after the real fill; and the side
reader uses `depth` pread threads instead of a ring when the pool does (`LLAMA_EXPERT_STORE_PREFETCH_URING`,
default: as the pool). Legs: production args, cache 424000 MiB, HBM tier, snapshot restored (67647 experts, 47 s)
before every leg, 3 x 800-token coding runs with `cache_prompt: false`, md5 ac78af0e5d on every run of every leg.

| leg | tok/s runs 1 / 2 / 3 | real fill per layer | predicted records per post, used | hit rate |
|---|---|---|---|---|
| base (deployed config, same binary, prefetch off) | 13.74 / 13.84 / 13.84 | 0.14 ms, 0.2 records | | 99.1 % |
| v2, CPU slot maps, pread threads, depth 8, `PREFETCH_INPUT=1` | 11.37 / 11.23 / 11.28 | 0.27 ms, 0.2 records, between fills 0.73 -> 0.78 ms | 1.27, 66 % | 98.7 % |
| same through one io_uring ring | 11.04 / 10.93 / 10.91 | 0.25 ms | 1.27, 62 %, adopt waits 1.2 ms | 98.7 % |
| v2 with the slot maps on the GPU (async hook) | 11.51 / 11.40 / 11.39 | | 1.27, 65 % | 98.7 % |
| v2, `PREFETCH_MAX=1`, `PREFETCH_DEMOTE=1` | 11.57 / 11.44 / 11.44 | 0.25 ms | 1.00, 41 % | 98.7 % |
| v2, `PREFETCH_MAX=2`, `PREFETCH_DEMOTE=1`, depth 16 | 11.52 / 11.42 / 11.46 | 0.25 ms | 1.21, 40 % | 98.7 % |

Why it loses here (and wins on the Spark): at 99 % residency the real misses are the tail of the routing
distribution, and so are the predictor's errors, so most predicted misses are wrong reads (1.0-1.3 records per
layer against 0.2 real misses) that each evict a resident expert when planned (the LRU is full; demoting the record
afterwards does not bring the victim back): hit rate 99.1 -> 98.7 %. The speculative reads also lengthen the real
fills (0.14 -> 0.25 ms per layer: records adopted in flight take 0.68-0.78 ms through 8 threads, a pool read 0.58
through 32, and the disk is busy when the real fill arrives), and the predictor itself costs ~0.05 ms of GPU time
per layer (between fills 0.73 -> 0.78 ms, ~4.6 ms per token), which alone is more than the ceiling of the gain:
the real reads are 0.14 ms x 92 = 13 ms of the 73 ms token, and a perfect predictor could hide at most that.
Net: 13.84 -> 11.3-11.5 tok/s, every variant. **Verdict: dropped on the GH200**, the engine support stays behind
the flag (it is what any box without io_uring or with the CPU split needs). Not worth revisiting unless the miss
rate rises (a smaller cache) or the predictor's GPU cost disappears.

Side result: with the snapshot restored and `cache_prompt: false`, run 1 of the deployed config is 13.74 tok/s,
not the 10.8 of the previous ledger entry (whose run 1 went through the prompt cache).

### Revisit when the bottleneck moves

Neutral is not dead: these are lossless and stay in the engine behind a flag, but measured neutral only because
something else dominated at the time. Re-test them after any change that shifts the per-token breakdown.

| change | why it was neutral | re-test when |
|---|---|---|
| Predictive prefetch (`LLAMA_EXPERT_STORE_PREFETCH=1`) | Spark: v2 (side reader, bounded, preemptible) is deployed, +12 %. GH200: v2 measured 2026-10-22 under the CPU split, -18 %: wrong guesses evict resident experts at 99 % residency and the predictor costs more GPU time than the reads it can hide | GH200: only if the miss rate rises (smaller cache) or the predictor becomes free |
| Hopper Q6_K TMA GEMV (on by default) | -1.6 ms/token in the op profile, but end to end within noise while miss waits dominated | after miss waits shrink further |
| `GGML_CUDA_GRAPH_OPT=1` | parallel attention branches hidden behind the expert waits | GPU-bound phases (fewer misses, prefill) |
| Async host-bank hook / CUDA graphs | launch overhead was not the bottleneck | when GPU time per layer dominates |
| Eviction policy (ARC / LFU decay, simulated) | ~8 % fewer misses at the GH200's capacity | tighter caches (the Spark's 10 %) |
| Hugepage destinations for O_DIRECT (3.9 vs 4.5 ms per fill) | THP on host regions gave NaN on the GB10 | once that NaN is understood |
| io_uring on the GH200 | one submitter = one virtio queue | with multiple rings across CPUs |

### Open ideas (not yet measured)

- **A more precise predictor for the speculative prefetch.** 70 % of the started reads are used; the window is full, so precision is the whole remaining gain on the Spark read side (see the prefetch v2 section).
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
