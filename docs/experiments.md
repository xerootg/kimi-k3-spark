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

## Current state (2026-10-22)

| | Spark | GH200 |
|---|---|---|
| image | g1989679 -> the split build (see the Spark split section) | g5fd3e3f (valve) -> the prefill lookahead build |
| decode, coding prompt, warm | 1.93 tok/s (io_uring reads, self-prefetch, speculative prefetch v2 with predictor input 3, split expert GEMV, snapshot, memory locked, NVMe IOMMU in passthrough) | 13.65-13.84 tok/s (cache 424000 MiB, slot maps on the CPU, snapshot restored at start; run 1 13.7 tok/s with `cache_prompt: false`) |
| cache | 52000 MiB host (hit rate ~60 %), memory valve keeps 4 GiB of MemAvailable | 424000 MiB host + 24 GiB HBM tier (hit rate ~99 % warm), memory valve keeps 4 GiB host / 6 GiB HBM free (releases ~5 GB at start) |
| prefill, 16k-token prompt | (copy path, not measured here) | 127-145 tok/s with the prefill lookahead (98-106 without), read-bound at the drive's 9.6 GB/s |
| where a token's time goes | ~0.52 s: per layer, ~2.8 ms of reads with the shared expert, the resident experts' matmuls and the predictor under them (the GPU still waits ~1.2 ms), then the misses' matmuls, the merge, routed_up and the next layer's attention (~2.3 ms) with ~4.5 predicted records read under them (48 % of them used) | ~105 ms: expert matmuls over C2C, Q6_K spine, miss waits |

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
| Spark + nuc2/nuc3 (RAM tier, CPU / Arc SYCL / NPU expert compute) after the Thunderbolt mesh, 2026-10-26 | mesh does not reach the Spark (2.5 GbE); predicted 0-1 % bit-identical, 3-6 % with other kernels; Arc = CPU speed (~23 GB/s IQ1_S); NPU needs Q4 requant | rejected (study section below) |
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

### Spark: precision of the speculative prefetch predictor (2026-10-21)

The question was how much of the drive-idle window is spent on records the real fill needs, and what a better
predictor would buy. Measured offline first: `tools/k3-pfdump` (in patch 0003) decodes a chat prompt greedily with an
eval callback that dumps, per decode token and layer, the predictor's 896 scores, the real router's selection scores
and the 16 real ids; the evaluator `tools/pfeval.py` scores predictors by rank, layer
and score gap, and replays the pipeline through an LRU of 7980 records (the snapshot's resident count) with a window
of 5 records per layer. The replay reproduces production: 6.7-6.8 misses per layer without speculation, 4.2-4.3 in
the self-prefetch fill with the deployed predictor (measured 4.1), 4.5 predicted records started per layer (4.8).

Findings, 400 tokens of the coding prompt and 300 of a prose prompt (same picture on both):

- The deployed predictor (`PREFETCH_INPUT=1`, the next router on the un-normed FFN input plus the shared expert) has
  top-16 precision 0.61-0.64, falling from 0.95 at rank 1 to 0.29 at rank 16; top-32 recall 0.76-0.79. Of the started
  speculative reads 58-59 % are used.
- The window is not the limit: on average 4.5 predicted misses fit in a window of 5, so ordering the predictions by
  score gap, absolute score or anything else changes nothing (all gating policies replay identically). The previous
  token's ids are no signal either: 32 % of them are in the real set, and of those not already predicted only 13 %.
  The limit is recall: the window can only be filled with better predictions, not more of the same ones.
- The model's cross-layer residual attention explains the worst layers. Every 12th layer banks the stream and restarts
  it from the attention output, and every layer's FFN input is a learned mix of the banked checkpoints and the stream
  (`res_mix`). The deployed predictor ignores the mixing, so on the checkpoint layers it scores 0.34 and on layer 13
  0.01 (its input is the restarted stream of layer 12, which looks nothing like the mix layer 13 applies).
- Variants built in the graph behind `LLAMA_EXPERT_STORE_PREFETCH_DUMP=1` (named tensors, same generation,
  measured in one run): the stream before this layer's FFN plus the shared expert, put through the *next* layer's
  own residual mix (on a checkpoint target: the stack plus this stream as the new checkpoint, the unknown attention
  output as zero) raises the precision to 0.675 (checkpoint layers 0.48, the layer after them 0.58, layer 13 0.46);
  adding this layer's whole routed output (an upper bound for computing the resident experts early) gives 0.717; the
  same two layers ahead 0.55 / 0.57.
- Replay with those: real fill 4.26 -> 3.94 records with the mixed input; 3.69 with the routed output but only if it
  had the whole window, 4.10 with the 3.5-record window left after the expert matmuls; two layers ahead in the spare
  window 3.88; skipping the checkpoint layers 4.08 (even their poor predictions help). Oracle 2.39.

So the mixed input (`LLAMA_EXPERT_STORE_PREFETCH_INPUT=2`, patches/README.md) was built for real: `res_mix_pred`
mixes the stream plus the shared expert with the next layer's `ffn_res_score` (plus the stream as an extra checkpoint
on checkpoint targets), the next FFN norm, the next router; a few small ops per layer, decode only (n_tokens <= 4).

A/B, `~/spark-bench.sh`, 3 x 200-token coding runs per leg after a 32-token warm-up, legs interleaved, md5 `e801484d`
on every run and `b2c79009` on every warm-up:

| leg | decode tok/s (runs 1/2/3) | predictions used of queued | self-prefetch fill |
|---|---|---|---|
| input 2 | 1.854 / 1.855 / 1.844 | 52.5 % | 3.8 records, 2.80 ms |
| input 1 (deployed) | 1.812 / 1.810 / 1.817 | 48.2 % | 4.1 records, 2.94 ms |
| input 2 again | 1.843 / 1.838 / 1.848 | 52.3 % | 3.8 records, 2.80 ms |
| input 1 again | 1.826 / 1.809 / 1.794 | 48.1 % | 4.1 records, 2.96 ms |

**Verdict: kept, deployed as `LLAMA_EXPERT_STORE_PREFETCH_INPUT=2` (image g1989679): 1.81 -> 1.85 tok/s (+2 %),
bit-identical**, the replay's +4 % minus the ops the mix adds to the window. What is left on this path is the next
layer's attention, which no input available before it runs can supply; the oracle replay (2.39 records) is the
ceiling and the resident experts' routed output (0.717) the last measurable step toward it, see "Open ideas".

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

### Spark: the NVMe behind the IOMMU in passthrough (2026-10-22)

DGX OS boots the Spark with `iommu.passthrough=0` (`/etc/default/grub.d/iommu.cfg`), so the NVMe's DMA ran in the
SMMU's `DMA-FQ` mode: every O_DIRECT piece is mapped and (lazily) unmapped per page. `iommu.passthrough=1` (a later
drop-in, `/etc/default/grub.d/zz-kimi-iommu.cfg`, then `update-grub` and a reboot) puts the NVMe's group
(`0004:01:00.0`, iommu group 13) in the `identity` domain; the GPU's group stayed as it was. The pod came back on
its own with the snapshot restored (7978 experts, 4.3 s, 12.3 GB/s).

| measurement (same binary and config: image g1989679, predictor input 2) | before | after |
|---|---|---|
| `readbench` 7-record fills, io_uring 1 MiB pieces, qd 64 (mean / p50 ms, GB/s) | 3.94 / 3.90, 11.7 | 4.10-4.13 / 4.10, 11.2-11.4 (three repeats) |
| same with 128 KiB pieces | 3.91 / 3.90 | 4.29-4.35 / 4.18-4.33 |
| same through the pread pool (16 threads) | 4.55 / 4.51 | 4.49 / 4.47 |
| 4-record fills, io_uring 1 MiB | 2.32 | 2.52 |
| decode, 3 x 200 tokens (md5 e801484d every run) | 1.854 / 1.855 / 1.844 and 1.843 / 1.838 / 1.848 tok/s | 1.866 / 1.876 / 1.853 tok/s |
| self-prefetch fill per post | 3.8 records, 2.80 ms | 3.9 records, 2.89 ms |

Verdict: **no change in the engine** (+1 % on the medians, inside the run-to-run spread; the fill costs the same
0.74 ms per record), while the fill microbenchmark got 4-5 % slower on io_uring and stayed put on the pread pool.
The per-page map cost the readbench hugepage comparison pointed at (3.9 vs 4.5 ms) is therefore not where the
identity domain helps; whatever `DMA-FQ` cost is left is below the noise of a decode. Kept (one code path less on
every fill, nothing lost), easy to revert by deleting the drop-in and rebooting.

### Spark: split expert GEMV, the resident experts' matmuls under the fill wait (2026-10-22)

The last item of the Spark read-side list (Open ideas, "the resident experts' matmuls under the fill wait"). Per
layer the GPU waited ~2.8 ms for the misses and only then ran the three `mul_mat_vec_q` over the 16 selected
experts (0.6 ms), although ~12 of the 16 were in the bank before the fill started. Built as
`LLAMA_EXPERT_STORE_SPLIT=1` (patches/README.md "Split expert GEMV"): the graph looks the experts up twice. The
early lookup returns after the plan with the slots of the complete experts (the misses point at a zero record,
one extra slot per group), and the misses' reads start on a fill thread; the shared expert, the early matmuls
over 17 ids (one scratch row) and the next layer's prefetch predictor run under the fill. The real lookup waits
for the fill and returns the misses' slots packed into `LLAMA_EXPERT_STORE_SPLIT_LATE` (default 8) rows; a
second matmul computes them and `ggml_set_rows` copies each miss's row into the early output at its position.
Every row is computed once, from the same slot, by the same kernel, and the weighted sum keeps its order: the
gold md5 held on all 21 runs of the A/B. A layer with more misses than late rows reads the excess at the early
lookup, synchronously (12 % of the layers, 0.14 records per layer on average), so the late matmul never
overflows. The zero record's rows are not free: the GPU reads the host bank uncached, so every dummy id costs
~28 us over the three matmuls, which is why a 16-row late matmul (all hits as dummies) gave back a third of the
gain and the late matmul is kept short.

Same binary and config for every leg (union tree at kimi-k3-spark 3991384 plus this change, predictor input 2,
io_uring, memory lock refused by the native shell's `RLIMIT_MEMLOCK` of 16 GB, which is why some runs stalled on
a prompt (marked *, a swapped page served behind the reads: see "Spark stalls"); interleaved legs of 3 x 200
tokens, warm-up md5 b2c79009 and run md5 e801484d every time:

| leg | tok/s runs 1 / 2 / 3 | split GEMV stats (per layer) |
|---|---|---|
| split off (baseline), two legs + the post-reboot run | 1.836 / 1.643* / 1.850; 1.651* / 1.549* / 1.844; 1.866 / 1.876 / 1.853 | self-prefetch fill 3.9 records, 2.85 ms |
| split, late matmul 8 rows, two legs | 1.937 / 1.930 / 1.905; 1.619* / 1.908 / 1.914 | 11.6 of 16 complete at the early lookup, 4.45 read under the early matmul, 0.14 read synchronously, wait at the real lookup 1.39 ms |
| split, late matmul 16 rows (no synchronous excess) | 1.911 / 1.917 / 1.898 | 11.3 complete, 4.71 under the early matmul, wait 1.50 ms |
| split, 8 rows, predictor input 3 (the early output's share of the routed output added), two legs | 1.935 / 1.850* / 1.949; 1.521* / 1.625* / 1.927 | precision 48.4 % of the predicted reads used vs 47.3 % with input 2; wait at the real lookup 1.22 ms |

Clean runs (no stall): baseline median 1.851 tok/s, split 1.914 (**+3.4 %**), split with 16 late rows 1.911,
split with input 3 1.935 (+1 % over input 2, the three clean runs all at or above input 2's best; precision 48-49 vs 47 %). The early lookup's plan costs 0.19 ms per layer on the poller (0.002 ms without
the synchronous excess), the GPU posts the real lookup ~1.4 ms into the ~2.8 ms fill and waits the rest, then
runs the late matmul (4.5 real + 3.5 dummy rows, ~0.25 ms) and the merge. What is left on this path: the misses'
own matmuls still run after the last record lands; a per-id wait inside `mul_mat_vec_q` (each block spinning on
its record's ready flag) would hide all but the last record's share, ~0.2 ms per layer (~3 %), at the price of a
kernel change. Verdict: **kept, deployed** (`LLAMA_EXPERT_STORE_SPLIT=1`, 8 late rows, predictor input 3).

### Both boxes: the memory pressure valve (2026-10-22)

Why: the GH200 pod was OOM-killed on 2026-10-22 04:29 UTC while prefilling a 30k-token prompt from the RE
workbench (node 0 at its 2 GB min watermark with 441 GB anonymous; HBM node ~4.7 GB free against a compute buffer
that grows ~10 GB on long prompts). The cache size had only ever been checked with 800-token decodes. The config
was shrunk to 416000 MiB + 14 GiB tier as a stopgap the same morning; the valve (patches/README.md, "Memory
pressure valve") makes the cache size itself instead. GH200 runs: production args, snapshot restored, 800-token
coding runs with `cache_prompt: false`, gold `ac78af0e5d`. Spark runs: `spark-bench.sh` with the deployed env
(split GEMV, prefetch input 3), gold `b2c7…` / `e801…`.

| what | result | verdict |
|---|---|---|
| First build, GH200 (416000 + 14 GiB, 8 GiB host reserve) | every `MADV_DONTNEED` failed with EINVAL except the few runs that happened to start on a 64 KiB page boundary, and those rounded the end up into the next, still resident, slot: md5 `846b7cdb92` / `0d6163e584` (wrong) from run 1, and since no page came back the valve kept releasing (123 GB released while MemAvailable stayed at the reserve) | fixed: only whole pages inside a run are dropped; a release that gives back under half of what it took stops the valve |
| Registered memory and `MADV_DONTNEED` on the GH200 (1 GiB test buffer: plain, `mlock2` on fault, `cudaHostRegister`ed, registered then unregistered; drop, CPU rewrite, GPU sum) | the drop succeeds in all four and the GPU sees the rewritten bytes: the host bank is read through the host page tables, registration does not pin a stale copy | no unregister needed |
| Fixed build, GH200 (416000 + 14 GiB, 4 GiB host reserve): no pressure / 12 GiB host hog / 14 GiB HBM hog / after re-arm | md5 `ac78af0e5d` on all four; host hog: 8.7 GB released in 3 batches, lowest MemAvailable 1.7 GB, no kill, all re-armed within a minute of the hog exiting; HBM hog: 7.5 GB of the tier released (stopped at the 50 % floor, HBM node free 2.2 GB at worst), all re-armed 50 s after it exited | works |
| Long-prompt gate, GH200: two 66k-token prompts at once | prefill ran at 13-33 tok/s (two slots of 2048-token ubatches through the bank), so the gate was stopped after 55 min at 68 % / 56 %; until then 3.7 GB of host cache released and re-armed, no kill, peak server RSS 435 GiB | partial; the long-prompt prefill speed on the GH200 is its own open item |
| Prefill scan resistance with released slots (GH200, 424000 + 24 GiB) | 11.01 / 11.05 / 10.99 tok/s with the valve vs 13.62 / 13.72 / 13.71 without: a released slot lowered `n_filled`, the LRU then took the cache for not full and switched the prompt's scan resistance off, so every prompt churned the working set. Released slots now count as full (`n_parked`) | fixed: 11.87 (run 1, right after the start-up release) / 13.86 / 13.84, md5 `ac78` x3 |
| GH200 at 424000 + 24 GiB with the valve (the sizes that OOM-killed the pod) | at start the valve releases 2.9 GB, then 2.2 GB more over three runs; MemAvailable stays 4.3-5.2 GB, HBM node free 6.5 GB | deployed: the cache sizes itself |
| Spark, valve on vs off, no pressure (split build) | 1.946 / 1.944 / 1.956 vs 1.936 / 1.953 / 1.932 tok/s, gold md5 on every run, nothing released (MemAvailable 8.6 GB) | neutral |
| Spark, 7 GiB hog during the runs | 5.6 GB released in 3 batches (MemAvailable 4.2 -> 2.6 GB at worst), 1.94 / 1.87 / 1.90 / 1.89 tok/s under it, gold md5 on all 5 answers, all re-armed 10 s after the hog exited | works |
| PSI trigger | `/proc/pressure/memory` accepts a trigger only from root on these kernels (EINVAL for the user, any window); the pods run as root, the native tests fell back to the 500 ms timer | timer is enough so far |

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
| Q6_K GEMV for Hopper: TMA-staged weight tiles, stock math (tools/q6k-gemv-bench.cu) | isolated: 2.45 -> 2.88 TB/s (output head 3.39), bitwise identical on 45 shape x seed cases; in the server unfused Q6_K 20.8 -> 19.2 ms/token, fused gate/up 5.4 -> 5.7 ms; end to end within noise (9.48-9.74 tok/s); re-tested 2026-10-26 at 13.5 tok/s: +2 % (see that section) | kept for unfused GEMVs |
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

### GH200: long-prompt prefill, the next layer's misses read under the current layer (2026-10-22)

The RE workbench sends 15-45k-token prompts, and its history condenser re-sends the whole context after every
condensation, so prefill speed is what the user waits on there. Production pod logs (Loki, 2026-10-22 03:00 UTC):
~100 tok/s for 6-18k-token prompts, ~20 s per 2048-token ubatch. (The 12 tok/s seen during the valve test was that
test: its first valve build had released half the cache and two 66k prompts ran at once.) A new per-ubatch stats
line (`expert store: prefill: ...`, every 4 ubatches) split the time: 92 batch fills per ubatch, each reading
133-145 records (the ~18 % of a layer's experts that are not resident; a 2048-token ubatch touches all 896 of them)
in 93-101 ms, synchronously, while the GPU waits. So ~9.2 s of reads + ~11 s of GPU work per ubatch, in series.
The drive reads at 9.6 GB/s, its ceiling, but only half the time.

Change: `LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=1` (CPU split only). After the real fill of layer il of a large batch
(8 ids per expert or more, `LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD_MIN`), the engine plans every expert of layer il+1
(scan resistant like any prefill fill; the victim search steps over layer il, which the GPU computes meanwhile) and
reads the misses on the fill thread. The next fill waits for those reads and finds them as hits. Under the CPU split
the lookup of il+1 cannot run before the GPU has finished il (its router input comes from il), so the slots of
il-1 and older are free; the memory valve leaves the layer in flight alone (protected, as the split GEMV's).

16k-token prompt (14257 tokens, a source file, `cache_prompt: false`), production args, cache 424000 MiB + 24 GiB
HBM tier + valve, the same snapshot restored before every leg, two windows:

| leg | prefill | per batch fill | md5 of the 48 answer tokens |
|---|---|---|---|
| A, lookahead off, run 1 (window 1 and window 2) | 97.8 / 98.5 tok/s (145 s) | 133-145 records read in the fill, 93-101 ms | 470f605de6, 470f605de6 |
| A, run 2 (same server, cache state after run 1) | 105.7 tok/s | 142 records, 98 ms | 2ac21c63c8 |
| B, lookahead on, run 1 (window 1 and window 2) | 127.2 / 126.8 tok/s (112 s) | 6-15 records in the fill (4-11 ms), ~190 read ahead, 41-45 ms waited for them | 39a978363b, 39a978363b |
| B, run 2 | **144.8 tok/s** | 4.5 records, 3 ms | 2ac21c63c8 |
| C, lookahead + ubatch 4096 | 185.9 tok/s (77 s) | 17 records, ~200 ahead | 8fa139d05c |

800-token coding run (`cache_prompt: false`) after the 16k prompt: md5 ac78af0e5d in A, B and C (10.43 / 10.56 / 9.88
tok/s, a first run after a restart).

**Prefill output depends on the cache layout, in the deployed engine too.** The same build, prompt and snapshot
gives the same text in two separate servers (A run 1 = 470f605de6 twice, B run 1 = 39a978363b twice), but a second
run of the same prompt in the same server, after the first one has changed which slots hold which experts, gives
another (A run 2 = 2ac21c63c8), and with the cache in that state the lookahead build gives the identical text (B run
2 = 2ac21c63c8). The lookahead changes where the misses land (it cannot evict the layer the GPU is computing), so
its run 1 differs from A's run 1 the way A's run 2 does. The texts are equivalent answers that diverge after a few
dozen tokens. Decode is unaffected (ac78af0e5d everywhere, and it never depended on the cache state). The source of
the layout dependence is not pinned down; the suspect is the MMQ `mul_mat_id` over the bank, which walks the
experts in slot order (the compacted expert list), so where a stream-k CTA boundary splits a tile's K range, and
the fixup's summation order, can depend on the layout. For prefill changes, "bit-identical" therefore means: the
same text from the same cache state, which is what B run 2 shows.

Ubatch 4096 (leg C, +46 % over B run 1) doubles the compute buffer (3.2 -> 6.3 GB of HBM): the valve released
half of the HBM tier (1887 slots, its floor) and HBM free still went down to 1.8 GB during the 16k prompt.
Too close for a box that must not croak; not deployed (see Open ideas).

**Verdict: kept, deployed on the GH200** (`LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=1`): prefill +29-37 % (98-106 ->
127-145 tok/s). Prefill is now read-bound with the drive busy nearly all the time (~190 records x 92 layers =
~118 GB per ubatch, 12.3 s at 9.6 GB/s, of ~14 s per ubatch); the GPU part (~11 s per 2048 tokens) is hidden
under it. Off on the Spark (it needs the CPU split; the Spark's prefill goes through the copy path anyway).

### GH200: the Hopper Q6_K TMA GEMV end to end, re-tested at 13.5 tok/s (2026-10-26)

The kernel has been in patch 0003 and on by default on the GH200 since the third pass (342e88c, image g342e88c);
it measured "within noise" end to end then, at 9.5 tok/s with miss waits dominating, so it sat in the revisit
list. Now that the cache holds ~98 % of the decode lookups, it was A/B'd again.

What it is (details in `patches/README.md`): `mul_mat_vec_q6_K_tma<R, has_fusion>` in `mmvq.cu`, a persistent
kernel that streams tiles of 2 or 4 weight rows into a 2-stage shared-memory buffer with TMA bulk copies, keeps
the q8_1 vector in shared memory, and runs the unchanged `vec_dot_q6_K_q8_1` with the stock thread mapping and
reduction order. Taken from the Q6_K case of `mul_mat_vec_q_switch_type` only for compute capability 9.x, one
column, no expert ids, one channel and sample, contiguous rows, >= 2048 rows, 16-byte aligned; the fused gate/up
stays on stock. `GGML_CUDA_MMVQ_Q6K_TMA=0` turns it off.

Isolated (H100, `tools/q6k-gemv-bench.cu`, K3 decode shapes, two runs): 7168x12288 29.5 -> 25.1 us (2.45 -> 2.88
TB/s), attn_output 30.0 -> 25.7 us, dense ffn_down 80.4 -> 67.4 us, output head 376 -> 285 us (3.39 TB/s); the
small shapes (attn_q_a, kv_a_mqa, ssm_f_a) are unchanged. Summed over one token's Q6_K GEMVs: 19.3-19.5 -> 16.9-17.0
ms if every shape took it (~2.4 ms). Bit-identical: 0/45 differing outputs in the ggml bitcheck (shapes x seeds);
test-backend-ops MUL_MAT and MUL_MAT_VEC_FUSION pass. Floor estimate: memory-only and compute-only versions of the
kernel each take ~22 us on the 7168x12288 shape, so what is left is their overlap plus ~2 us of launch start and
drain.

Variants that failed (all bit-identical, 7168x12288, stock 29.5 us, kept variant 25.1 us):

| variant | result |
|---|---|
| register lookahead of 2 / 4 blocks on the stock kernel | 37.6 / 48.5 us |
| L1-bypass and evict-first hints on the stock loads | 47-48 us |
| 2 / 4 rows per block on the stock kernel | 28.5-29.0 us |
| 3, 4 or 8 TMA stages | up to 34.9 us |
| 8 blocks per SM; tile rows split across warp groups | 26.8-36.6 us |
| mbarrier instead of `__syncthreads` | 25.2 us |
| aligned 32-bit loads + funnel shifts, hoisting the -32 term | no gain |

End to end (same native build of the 9eefe4f patch union in `~/llama-k3-la`, production args and env, the
production snapshot frozen at the pod's shutdown and copied fresh before each leg, 3 x 800-token coding runs with
`cache_prompt: false`, legs A1 B1 A2 B2; `~/q6k-ab.sh`, log `~/q6k-ab.log`):

| leg | run 1 | run 2 | run 3 | md5 (all runs) |
|---|---|---|---|---|
| A1 `GGML_CUDA_MMVQ_Q6K_TMA=0` | 9.59 | 12.39 | 13.45 | ac78af0e5d |
| B1 default (TMA on) | 9.67 | 12.57 | 13.63 | ac78af0e5d |
| A2 `GGML_CUDA_MMVQ_Q6K_TMA=0` | 9.60 | 12.61 | 13.51 | ac78af0e5d |
| B2 default (TMA on) | 9.77 | 12.99 | 13.79 | ac78af0e5d |

Hit rate 98.4 % in every leg. B beats A in all six paired runs: steady runs 2-3 average 12.99 -> 13.25 tok/s
(+2.0 %), run 3 13.48 -> 13.71 tok/s (74.2 -> 72.9 ms/token, ~1.3 ms saved, close to the 1.6 ms the op profile
showed in the third pass; the fused gate/up and the small shapes keep the rest of the isolated 2.4 ms). Run 1 (cold
GPU-side after the restore) 9.60 -> 9.72.

Production check after the window, over the API on an idle server: 800 tokens at 9.76 tok/s, md5 ac78af0e5d. (A
first check while another client's 66k-token prompt shared the batches gave 18d98db589: mixed batches take other
kernels, so the reference check needs an idle server as well as `cache_prompt: false`.)

**Verdict: kept, on by default on the GH200 (no deploy needed: production image g9eefe4f already runs it).** The
GB10 keeps it opt-in (`=2`, not bit-identical there, see the third pass).

### Spark + the Intel NUCs over the new Thunderbolt mesh: study and micro-benchmarks (2026-10-26)

The question: with the new Thunderbolt mesh between nuc2, nuc3 and nas-large (argo-things `configs/tbnet`), does
spreading the Spark's work onto the NUCs' RAM, CPUs, Arc iGPUs or NPUs get anywhere near reasonable? Scope set
by the user during the study: nas-large is storage only (no compute, no RAM tier); compute candidates are nuc2 and
nuc3. Quality ranking: (1) bit-identical first, (2) same weights with other kernels (measure KL), (3) requantized
weights last. Nothing deployed; the Spark pod was not touched.

**Inventory (measured).**

| | nuc2 | nuc3 | nas-large | Spark |
|---|---|---|---|---|
| CPU | Core Ultra 5 225H (4P + 8E + 2LPE) | same | i5-1235U | GB10 (10 X925 + 10 A725) |
| RAM | 62 GB DDR5-5600, 2 channels; ~33 GB available, the Qwen3-Coder `llama-cpp` pod requests 60Gi | 30 GB, **one DIMM** (single channel); ~11 GB available (GitLab, Prometheus, Jellyfin, ...) | 62 GB DDR5-4800 (excluded) | 121 GB, ~8 GB available with the pod up |
| iGPU | Arc 130T (8086:7d51, 7 Xe cores), i915 | same silicon, reports as "Intel Graphics" (8086:7dd1: no Arc branding on one channel) | Iris Xe (excluded) | |
| NPU | NPU 3720 class (8086:7d1d), `intel_vpu` loaded, `/dev/accel0`, no user-space driver installed | same | | |
| Network | 2.5 GbE + Thunderbolt to nuc3 (20 Gb/s link) and nas-large (40 Gb/s link) | 2.5 GbE + TB to nuc2 and nas-large | 2.5 GbE (VLAN 3) + 10 GbE (VLAN 1) + TB | **2.5 GbE only** (Realtek RTL8127, a 10GBASE-T chip negotiating 2.5G; no ConnectX, no USB4/Thunderbolt) |

Network (Python TCP bench `tools/distributed/netbench.py`): Thunderbolt pairs 14-18 Gbit/s (1.75-2.2 GB/s) with one
or four streams, but small-message latency is not better than the LAN (14 KB request/response p50 0.16-0.31 ms,
ping 0.18-0.27 ms). The Spark reaches every box only over its 2.5 GbE port: 2.25-2.35 Gbit/s (~290 MB/s); 14 KB
each way p50 0.50 ms / p90 0.65 ms, 28 KB p50 0.60 / p90 1.0 ms. **The Thunderbolt mesh does not touch the Spark**, so
every per-token exchange with it crosses 2.5 GbE.

**Routing replay (`tools/distributed/remotesim.py`).** Real top-16 ids from the 2026-10-21 predictor dumps
(`pfdump-prose.bin` 300 tokens, `pfdump-code*.bin` 400 + 400), replayed through the Spark's 7980-slot LRU plus a
static remote set of the most frequent (layer, expert) pairs that is never cached locally. Trained on prose,
evaluated on code (an honest out-of-domain split); "mixed" trains on prose plus the other code prompt.

| remote set | GB (6.45 MB records) | NVMe misses per layer | remote experts per layer (mean / p90 / max) |
|---|---|---|---|
| none (production) | 0 | 6.80 | |
| 600 (what nuc3 could spare) | 4 | 6.73 (-1 %) | 0.21 / 1 / 6 |
| 3700 (nuc2 with Qwen3-Coder kept) | 24 | 6.14 (-10 %); mixed 5.82 (-14 %) | 1.27 / 3 / 14 |
| 7000 (nuc2 with Qwen3-Coder off) | 46 | 5.54 (-18.5 %); mixed 5.08 (-25 %) | 2.24 / 5 / 15 |
| 20000 (not available anywhere) | 132 | 3.52 (-48 %) | 5.38 / 9 / 16 |

Skipping the hottest 4000-7980 experts (the ones the LRU keeps anyway) changes almost nothing (within 0.1 miss).

**Remote expert compute, real records (`tools/distributed/moebench.cpp`).** One token, one layer's routed FFN
(gate/up `MUL_MAT_ID`, SwiGLU, down) on k experts drawn from 64 real store records (layer 1 IQ1_S, 413 MB; layer 11
IQ2_XXS gate/up), so the weights stream from DRAM. Upstream llama.cpp at LLAMA_SHA; CPU build in ubuntu:24.04 (the
`alderlake` variant loads), SYCL build with oneAPI 2026.1.1 (`intel/deep-learning-essentials`, upstream's pinned
image; 2025.0 is too old for this tree), compute-runtime 26.18 + IGC 2.34.4 user-mode driver extracted next to the
binary, run as the user with the render group (nothing installed on the hosts).

| backend (nuc2 unless noted) | k=1 | k=2 | k=4 | k=8 | k=16 | weight stream at k=16 |
|---|---|---|---|---|---|---|
| CPU, 4 threads | 0.57 ms | 0.89 | 1.68 | 3.19 | 6.31 | 16 GB/s |
| CPU, 8 threads (best) | 0.48 | 0.76 | 1.41 | 2.03 | 3.89 | 27 GB/s |
| CPU, 14 threads (LP-E cores on the critical path) | 0.84 | 1.10 | 1.75 | 4.35 | 8.31 | 12 GB/s |
| nuc3 CPU, 8 threads (one channel, busy box) | 0.90 | 1.29 | 2.14 | 4.27 | | 12 GB/s |
| Arc 130T, SYCL (any of ENABLE_OPT=0, PRIORITIZE_DMMV, MMVQ_WIDE=0, ESIMD=0: same or worse) | 0.62 | 0.77 | 1.29 | 2.34 | 4.49 | 23 GB/s |
| CPU, IQ2_XXS layer, 8 threads | 0.67 | 1.26 | 1.74 | | | 18 GB/s |
| Arc 130T, IQ2_XXS layer | 0.63 | 0.94 | 1.61 | | | 20 GB/s |

The iGPU is no faster than the CPU: the IQ1_S grid-decode kernels cap it at ~23 GB/s, a third of the
DDR5's bandwidth. It has tighter tails (p90 0.83 vs 1.08 ms at k=2).

Full round trip from the Spark (`tools/distributed/rrclient.c`, C, TCP_NODELAY; server = moebench `serve:` mode on
nuc2's CPU, 8 threads; request = k ids + 3584 f32, reply = k unweighted rows of 3584 f32): k=1 p50 1.34 / p90 1.77
ms, k=2 2.05 / 2.48, k=4 2.67 / 3.33, k=8 3.90 / 4.80; **k drawn from the 7000-set histogram (mean 2.7 when
non-zero): p50 1.88, p90 3.33, p99 5.76 ms**. That is ~0.5 ms over compute plus wire time (thread wake-up between
requests). A rerun with `OMP_WAIT_POLICY=active` was spoiled: nas-large went down for maintenance in the middle
(cordoned with kubectl at 04:45Z, NotReady 05:06Z, not caused by this study) and Longhorn started rebuilding
replicas on nuc2, so measurements stopped there. The SYCL round trip was never measured for the same reason.

**Model against the deployed Spark (1.93 tok/s = 5.6 ms per MoE layer).** The only part of the layer that fewer
NVMe misses can remove is the GPU's wait at the real lookup, ~1.2-1.4 ms per layer (split GEMV section): the
self-prefetch fill of ~3.9 records (0.72 ms each) minus the ~1.4 ms of GPU work under it. Removing every miss would
give ~2.46 tok/s; that is the ceiling for any shape that only removes reads.

| shape | quality | predicted tok/s | why |
|---|---|---|---|
| **A. NUC RAM as an expert tier, the Spark computes (bit-identical)** | identical | **~1.93-1.95 (0-1 %)** | 290 MB/s is 2.6 % of the NVMe's 11 GB/s; one record takes 22 ms over the wire, about 4 layers, so it can only carry predictions 4+ layers ahead (precision below 0.55). With a 10 GbE link to the Spark (its RTL8127 can do 10GBASE-T; needs a 10G switch port) it would be ~11 % extra bandwidth and ~5.5 ms per record: realistically +3-5 %, still bit-identical. |
| **B. NUCs compute the experts they hold (same weights, other kernels)** | KL not measured (not built) | **ideal 2.12-2.2 (+10-14 %); with the measured round trip ~2.0-2.05 (+3-6 %)** | A 7000-expert set on nuc2 (needs the Qwen3-Coder pod off nuc2) cuts misses 18.5-25 %: fill 3.9 -> 2.9-3.2 records, wait -0.5 to -0.7 ms per layer. But the rows must be back before the merge, ~2.5 ms after the router, and the measured round trip is p50 1.9 / p90 3.3 / p99 5.8 ms: ~30 % of the layers stall 0.5-3 ms, ~0.3 ms per layer on average, which eats half or more of the gain. No head start is possible: the experts need the real FFN input, which exists only right before the router. With nuc2 keeping Qwen3-Coder (3700 set): +2-4 %. |
| B', bit-identical remote compute | identical | not viable | The integer dot products and even the accumulation order of CUDA's MMVQ could be emulated on the CPU, but the fused SwiGLU uses the GPU's own `expf`; splitting it out means two round trips per layer (~3-4 ms), more than the window. |
| C. Whole layers on the NUCs (the 2026-10-18 RPC shape) | other kernels | ≤ 1.95 | nuc2 holds ~7 layers; all 16 experts of a layer remote = 3.9-4.5 ms compute + RTT in series with the Spark, which then idles. Same conclusion as the old RPC row. |
| D. NPU | requantized | not a fit | The upstream OpenVINO backend (the only llama.cpp path to the NPU) supports Q4_0 as its main NPU format and requantizes everything else. It has no IQ1_S/IQ2_XXS path, uses static graphs, runs one sequence and is stateless only. IQ1_S -> Q4_x is 2.9x the bytes (nuc2 would hold ~2400 experts instead of 7000; Q4_1 with 32-groups can hold IQ1_S's three levels per block almost exactly, Q4_0 cannot). It shares the same DDR5 and adds dispatch latency. Not measured beyond the backend's documented constraints. |
| nas-large as storage | n/a | no tok/s effect | Loading a 46 GB shard into nuc2 over Thunderbolt (~2 GB/s link) instead of the Spark's 2.5 GbE (~160 s): a load-time convenience only. The repacked store is not on nas-large; Garage there holds the GGUF on the HDD array. |

**Verdict: not worth pursuing.** Nothing bit-identical can win at 2.5 GbE. The best same-weights design is a
sizeable engine project: an asynchronous network expert path inside patch 0003, with the activation out of the GPU
mid-layer and rows merged through the split GEMV's `set_rows`. Even with nuc2 given up entirely it lands around
2.0-2.05 tok/s, maybe 2.1 with a tuned server. That is still under the user's bar. The binding constraints are the
Spark's 2.5 GbE port and the 2.5 ms window per layer, not the NUCs' compute or the mesh. KL/top-1 was not measured
because no design was worth building. What would change the picture: a 10 GbE (or faster) link to the Spark, for
shape A at a bit-identical +3-5 %; more memory on the Spark itself; or the GH200 path (13.8 tok/s).

### Spark: eviction policy on the 10 % cache, and CPU pinning (2026-10-26, measured)

The question was how much of the remaining per-layer wait (~1.2 ms of a ~5.6 ms layer, the fill of ~3.9 missed
records at the drive's ceiling) a better eviction policy could remove on the Spark, where the cache holds 9.7 % of
the experts and LRU hits ~58-60 %. The GH200 answer (2026-10-20: ~8 % fewer misses at 99 % residency) does not
transfer; the replay was redone at the Spark's capacity on the real top-16 selections of the two `k3-pfdump` traces
(400 decode tokens of the coding prompt, 300 of the prose prompt), one global cache of 7980 slots as `pfeval.py`,
scored after 150 tokens (`tools/cachepolicy.py`). The per-layer partition (86 slots
each) gives the same ordering within 0.5 points.

| policy | hits, code / prose (%) | with the deployed speculation, window 5 |
|---|---|---|
| LRU (deployed) | 57.50 / 61.66 | 73.36 / 76.41 |
| ARC | 58.51 / 61.75 | 73.03 / 76.14 |
| LFU, counts halved every 32k lookups | 58.89 / 64.33 | 70.36 / 74.79 |
| S3-FIFO (10 % small queue, ghost, 2-bit frequency) | 59.53 / 64.66 | 74.68 / 77.96 |
| CLOCK, 1 bit, hits do not move | 58.52 / 62.08 | 73.32 / 76.77 |
| **LRU + second chance, count <= 3** | **59.64 / 64.11** | **74.43 / 77.78** |
| Belady (needs the future) | 76.62 / 79.24 | |

Cache size on the same replay, LRU: 6000 slots 52.1 / 56.6, 9600 61.4 / 65.4, 12000 66.0 / 69.6, 16000 72.1 / 75.6
(code / prose): +20 % slots buys about as much as the best policy, and there is nothing left to free on the Spark
(`--cache-ram 0` already). LFU loses under speculation (wrong guesses never earn a count but a decaying LFU still
ranks old hot experts above them); S3-FIFO and the second chance both demote unused predictions by construction.
The second chance is one byte per slot in the existing list (patches/README.md), so it went in, as
`LLAMA_EXPERT_STORE_SECOND_CHANCE=3`. Expected on the Spark: ~5 % fewer misses per layer (4.26 -> ~4.05 records in
the self-prefetch fill), ~0.15 ms per layer, **~+2-3 % decode**; the 19-point gap to Belady is the same locality the
predictor already mines, no policy without the future closes it.

Functional check, granite-3.0-1b-a400m Q8_0 on the Framework's Intel iGPU (Vulkan, `-ngl 99 --n-cpu-moe 99`,
greedy, 160 tokens, same prompt): the text is identical to the no-store baseline in every run (md5 `33f95cd21f`);
at 128 MiB (79 slots, 10 %) LRU and second chance both read every expert (hit rate 0.00 %, no working set fits);
at 1024 MiB (641 slots, 83.5 %) LRU 96.60 % (1044 misses), second chance 97.04 % (909 misses, -13 %).

In the same build: `LLAMA_EXPERT_STORE_CPUS=<list>` pins the I/O workers, the speculative reader, the fill thread
and the CUDA host-bank poller. The GB10's X925 core ids from `/sys/devices/system/cpu/cpu*/cpu_capacity` are
5-9,15-19 (capacity 997-1024; the A725s are 0-4,10-14 at 718-731).

Spark A/B, measured 2026-10-26 (native build of this patch set off the pinned SHA, production args and env, snapshot
restored and copied fresh per leg, 3 x 200-token coding runs with request-level `temperature 0`). All 30 runs across
every leg gave decode md5 `e801484d` and warm-up `b2c79009`, bit-identical to the deployed build.

| leg | decode tok/s, runs 2-3 | hit rate |
|---|---|---|
| A baseline (knobs off), interleave A1/A2 | 1.920 / 1.929 ; 1.939 / 1.934 | 76.3 / 76.4 % |
| B `SECOND_CHANCE=3`, interleave B1/B2 | 1.957 / 1.970 ; 1.964 / 1.956 | 77.1 / 77.1 % |
| C B + `CPUS=5-9,15-19` (single leg) | 1.930 / 1.938 / 1.931 | 76.8 % |

Second chance: **+1.8 % decode (1.93 -> 1.96), +0.8 point hit rate (76.3 -> 77.1)**, reproducible across both
interleaved pairs, the self-prefetch fill 3.9 -> 3.8 records. Kept and worth deploying. CPU pinning (leg C, all
threads on the ten X925s) gave nothing over B and if anything a touch slower: the poller and the readers then
contend with the eight `--threads 8` matmul threads, which also want the big cores. Not deployed; a disjoint split
(readers on a couple of cores, matmul on the rest) is the only version worth another look, and only if it is free.

Long-context stability (the user's standing worry: >30k-token prompts OOM-crashing). One ~31k-token synthetic
prompt, single sequence, `--ctx-size 40960 --parallel 1`, valve on, `max_tokens 64`, baseline and with second
chance. Both completed, prompt 30947 tokens; no OOM, no crash, dmesg and server log clean; decode md5 `94ff1bf0`
identical between the two. MemAvailable low-water 7.18 GB (baseline) / 8.33 GB (second chance), both well above the
4.29 GB reserve, and the valve released zero slots in either: at this ctx the KV and compute buffers still fit
without evicting cache. Prefill ran at ~9.5 tok/s (56 min wall for the 31k prompt; `PREFILL_LOOKAHEAD` auto-off
here, it needs the CPU split and no split GEMV) and decode at 0.67 (baseline) vs 0.34 (second chance) tok/s over the
64-token tail -- a single short noisy sample at 31k context, not a real second-chance regression (the short-prompt
A/B and the bit-identical md5 rule that out), but long-context decode speed is unmeasured and worth a clean run if
it matters. The 2-sequence 2x66k case that OOM'd before (seventh pass) was deliberately not repeated.

### Revisit when the bottleneck moves

Neutral is not dead: these are lossless and stay in the engine behind a flag, but measured neutral only because
something else dominated at the time. Re-test them after any change that shifts the per-token breakdown.

| change | why it was neutral | re-test when |
|---|---|---|
| Predictive prefetch (`LLAMA_EXPERT_STORE_PREFETCH=1`) | Spark: v2 (side reader, bounded, preemptible) is deployed, +12 %. GH200: v2 measured 2026-10-22 under the CPU split, -18 %: wrong guesses evict resident experts at 99 % residency and the predictor costs more GPU time than the reads it can hide | GH200: only if the miss rate rises (smaller cache) or the predictor becomes free |
| `GGML_CUDA_GRAPH_OPT=1` | parallel attention branches hidden behind the expert waits | GPU-bound phases (fewer misses, prefill) |
| Async host-bank hook / CUDA graphs | launch overhead was not the bottleneck | when GPU time per layer dominates |
| Eviction policy (ARC / LFU decay, simulated) | ~8 % fewer misses at the GH200's capacity | tighter caches (the Spark's 10 %) |
| Hugepage destinations for O_DIRECT (3.9 vs 4.5 ms per fill) | THP on host regions gave NaN on the GB10 | once that NaN is understood |
| io_uring on the GH200 | one submitter = one virtio queue | with multiple rings across CPUs |

### Open ideas (not yet measured)

- **GH200 prefill past the read ceiling.** With the lookahead, a 2048-token ubatch reads ~118 GB (every
  non-resident expert once) at the drive's 9.6 GB/s; reads per token halve with each doubling of the ubatch
  (4096: 186 tok/s measured), but the compute buffer doubles too and the HBM has no room for it next to a 24 GiB
  tier. A smaller HBM tier plus ubatch 4096 is a trade of decode speed (the tier's hits) for prefill speed: measure
  both before choosing. Also unmeasured: posting layer 1 of the next ubatch at the last layer (~1 %).
- **Prefill numerics that do not depend on the cache layout.** See the GH200 prefill section: the same prompt gives
  a different (equivalent) text depending on which slots hold the experts. Pin down the kernel (suspect: MMQ
  stream-k over the slot-ordered expert list) and, if cheap, make it order-independent, which would make every
  prefill change testable bit for bit.
- **The misses' matmuls as their records land.** The split expert GEMV (2026-10-22) hides the resident experts'
  matmuls under the fill; the misses' still run after the last record. A per-id wait inside `mul_mat_vec_q`
  (each block spinning on its record's ready flag, set by the reader after the DMA) would hide all but the last
  record's share, ~0.2 ms per layer (~3 %), at the price of a kernel change and a mapped flag per slot.
- **A zero record the GPU caches.** The dummy rows of the split's late matmul cost real bandwidth (~28 us per id)
  because the GPU reads the host bank uncached; a zero record in device memory mapped into the same virtual range
  (CUDA VMM: `cuMemAddressReserve` + `cuMemMap`, host-NUMA allocations for the bank) would make them free and let
  the late matmul keep 16 rows without the synchronous excess.
- **A more precise predictor still.** Input 2 has precision 0.675; the rest is the next layer's attention, which is
  only known after it runs. The oracle replay (2.39 records in the self-prefetch fill vs 3.9) is the ceiling.
- **The rest of the Spark perf-hunt list.** Shared expert and routed_down under the I/O wait (self-prefetch with
  the real ids, est. +6 %), poller falling asleep inside the GPU gap (+2 %), I/O and poller threads on the X925
  cores (+2 %). The reads now run at the drive's ceiling, so more overlap is the remaining lever on the read side.
- **Why THP on the host regions gives NaN on the GB10.** Fixing it would bring back hugepage destinations
  (3.9 vs 4.5 ms per fill), but only if the pages also survive the load's reclaim.
- **Frequency-aware eviction for the host cache.** The routing skew that makes the HBM tier work should also
  help a policy like LFU or ARC beat plain LRU, which matters most on the Spark's 10 % cache.
- **Q6_K GEMV for Hopper.** The TMA kernel (kept, +2 % end to end) brought the big shapes to 2.8-3.4 TB/s; what
  is left is the fused shared-expert gate/up (still stock, the TMA version measured slower in the server) and the
  small shapes (attn_q_a, kv_a_mqa, ssm_f_a at 0.2-1.5 TB/s, launch-bound). Each variant listed in the 2026-10-26
  section is already measured; don't repeat them.
- **15+ tok/s on the GH200.** Only path found so far: all experts resident in memory, for example a second
  GH200 with expert parallelism.
