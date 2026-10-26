# Kimi K3 on one DGX Spark: what the llama.cpp patches change, and what each one bought

This is an overview for someone who has not followed the project. It explains why stock llama.cpp cannot run
Kimi K3 usefully on a DGX Spark, what the patches in [`patches/`](../patches/) change, and how much each change
helped. It also lists what was tried and dropped. Every number below comes from the experiment ledger
([`docs/experiments.md`](experiments.md)) or from the deployment notes it summarises. Each knob is described in
detail in [`patches/README.md`](../patches/README.md).

The same patch set is published as a branch of a llama.cpp fork:
[`xerootg/llama.cpp`, branch `kimi-k3`](https://github.com/xerootg/llama.cpp/tree/kimi-k3). The branch has one
commit per patch on top of the pinned upstream commit.

Short version: on a single Spark, decode went from **147 s per token** (unpatched llama.cpp) to **1.93 tok/s**
(the deployed configuration), and to **1.96 tok/s** with the newest change, which has been measured but is not
deployed yet. That is about 280 times faster. The output is bit-identical at temperature 0 at every step.

## The problem

**The model.** [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) is a 2.8T-parameter mixture-of-experts model.
It has 93 layers. 92 of them are MoE layers with 896 routed experts each, and every token uses the top 16 plus 2
shared experts. The attention is 69 KDA (linear) layers and 24 gated-MLA layers. The quantization used here is
Unsloth's `UD-IQ1_S` GGUF, which is 594 GB:

- **Routed experts:** about 545 GB, mostly IQ1_S, with IQ2_XXS parts in 12 layers.
- **Spine:** attention, the dense layer, the shared experts and the output head. About 45.5 GB on the GPU,
  mostly Q6_K.

**The box.** An NVIDIA DGX Spark has:

- a GB10 with 128 GB of unified LPDDR5X (about 273 GB/s peak), shared by the CPU and the GPU;
- one Gen5 NVMe, a Samsung PM9E1. Random reads over blocks that hold data run at 10.5-11 GB/s.

After the spine, the KV cache, the compute buffers and the prefill staging are in place, the expert cache gets
52000 MiB. That is 7980 expert records, or 9.7 % of all experts. The other 90 % have to come from the NVMe while the model runs.

**The arithmetic.** Each decoded token selects 16 experts in each of 92 layers. At about 6.6 MB per expert, that
is about 9.8 GB of expert weights per token. If all of it were streamed at about 7 GB/s, the floor would be about
1.4 s per token. Pulsar, an SSD-streaming engine built for this case, measured 0.73-0.75 tok/s on the same box,
which matches that floor. Beating the floor needs three things:

- a cache that serves many of the 16 × 92 lookups from RAM;
- reads that run at the drive's ceiling;
- reads that overlap with the GPU's work instead of alternating with it.

### Why unpatched llama.cpp ran at 147 s per token

Stock llama.cpp runs with `--load-mode mmap --n-cpu-moe 93`. Note that `--load-mode auto` resolves to `none` on
the GB10, because its CUDA device reports no mmap support. With `none`, llama.cpp reads all 545 GB of experts
into RAM until the OOM killer stops it, so `mmap` has to be forced. In mmap mode the experts stay file-backed and
the kernel page cache acts as the expert cache.

Every selected expert is then read through the page faults of the compute threads, which is slow for three
reasons:

- **Small synchronous reads.** Each fault reads one 4 KiB page synchronously.
- **No read-around.** Linux switches read-around off for a mapping after about 100 misses (`mmap_miss` in
  `mm/filemap.c`), so the reads stay at 4 KiB.
- **Reclaim on the fault path.** Every faulting thread also reclaims page cache itself. `allocstall` climbed by
  millions, and direct reclaim scanned 11 pages for every page it freed.

Measured result: **147 s per decoded token, and 1349 s to prefill 102 tokens.**

## The quality rule

No change is allowed to alter which experts the model uses or their weights. That rules out a lower
`expert_used_count`, requantization and approximate kernels. The check is **bit-identical output at temperature
0**:

- **Spark.** A fixed request sequence (a 32-token warm-up, then three 200-token coding answers) must reproduce
  the gold md5s on every run of every A/B leg: `b2c79009` for the warm-up and `e801484d` for each run.
- **GH200.** An 800-token coding run with `cache_prompt: false` on an idle server must give md5 `ac78af0e5d`.
- **Before new engine code reaches the Spark,** it runs on a laptop with a small MoE
  (granite-3.0-1b-a400m, Q8_0, Vulkan, the same `-ngl 99 --n-cpu-moe 99` topology). The output must be
  token-identical to the no-store baseline both with a 10 % cache and with an 83 % cache.

Lessons from applying the rule:

- **A wrong md5 has caught real bugs:**
  - misaligned expert blocks in the first GPU build (the output was `????`);
  - a valve build that zeroed the edge of a neighbouring slot;
  - an io_uring bookkeeping error in the speculative reader.
- **Reference runs must avoid the prompt cache.** A restored prompt-cache checkpoint whose KV was computed in a
  two-sequence batch is not bit-identical to one computed alone. A mixed batch with another client also uses
  other kernels. So reference runs disable the prompt cache and run on an idle server.
- **Prefill is checked differently.** Prefill output depends on which cache slots hold which experts, even in
  older builds. Prefill changes are therefore checked as "same cache state, same text". Decode output never
  depended on the slot layout.

Changes rejected on these grounds:

| change | speed | why it was rejected |
|---|---|---|
| 12 instead of 16 experts per token | 1.71 tok/s at the time | changes the model's output |
| n-gram speculation on the GH200 | 7.9 → 8.05 tok/s | not bit-identical |
| the Hopper Q6_K kernel on the GB10 | 5-10 % faster per kernel | not bit-identical on that chip |

## The improvements, in the order they were built

The A/B numbers below are decode speeds on warm caches. The early phases (0001 to the GPU host bank) used a
102-token prompt with 64-token answers. From the io_uring change on, the A/Bs used the fixed 3 × 200-token coding
sequence. Each "before → after" pair comes from the same A/B, with interleaved legs where noted. Baselines drift
a little between days, so neighbouring rows do not chain exactly.

### 1. `MADV_WILLNEED` prefetch of the selected experts (patch 0001)

**What it does.** Before anything touches the selected experts, the patch issues `madvise(MADV_WILLNEED)` for
the whole slice of every one of them. This turns the 4 KiB synchronous faults into large asynchronous reads that
overlap with the compute of the earlier experts. It hooks three places:

- the CPU graph start, for decode with `--n-cpu-moe`;
- the scheduler's expert upload before a prefill matmul is offloaded to the GPU;
- upstream's `--moe-cache-mib` fill.

**Effect.** 147 → 4.45 s per token (33×), and prefill of 102 tokens 1349 → 398 s. The volume of reads now matched
the ideal of about 8.4 GB per token, but at only about 2.2 GB/s, because the prefetch still reclaimed page cache
on its own path. Two attempts to fix that changed nothing: `MADV_COLD` after each matmul, and a larger
`vm.watermark_scale_factor`.

**Why it was not enough.** The page cache is the wrong expert cache. It works in 4 KiB pages, it puts reclaim on
the critical path, and it reads each expert as three scattered 2.2 MB slices, one per matrix. The patch is still
in the build: patch 0003 replaces its decode path, and on memory that is not file-backed it does nothing.

### 2. The expert store (patch 0003)

**The store file.** The core of the engine. [`tools/repack-experts.py`](../tools/repack-experts.py) rewrites the
routed experts into an **expert store** file:

- one 4 KiB-aligned record per (layer, expert);
- gate, up and down back to back inside each record, 6.45-7.83 MB per record;
- 545 GB in total.

The GGUF is still loaded for the spine.

**The cache.** The engine builds on upstream's `llama_moe_cache` (slot maps, per-group LRU, bank views that
`MUL_MAT_ID` reads), with these changes:

- **One read per expert.** The cache lives in host RAM on the CPU backend. Each group of layers with the same
  layout gets one populated region of records, and the gate, up and down banks are strided views into it, so a
  single read fills an expert.
- **Decode fills from a graph hook.** A hook on the CPU graph sees each layer's `GET_ROWS(slot_map, ids)` lookup.
  It plans the LRU and reads the misses with `O_DIRECT` before the matmul runs.
- **Prefill uses a staging area.** For large batches, each layer's used records are read once into a pinned
  staging area and uploaded from there.
- **The mapping is out of the expert path.** The loader no longer prefetches the 545 GB expert mapping.

**Effect.** 0.71 tok/s (1.40 s per token), up from 0.22. The hit rate was 55 % from a cache holding 10 % of the
experts: expert selection has strong temporal locality. The engine read 4.4 GB per token, and prefill of 102
tokens took 54 s instead of 398 s.

**The next bottleneck.** Instrumentation showed about 6.5 ms of reads per layer, at 9.6 GB/s for about 9
records, and **12 ms between fills**, with the CPU only a quarter busy. Moving from 16 to 8 matmul threads took
the GB10's little A725 cores off the critical path and gave 0.84 tok/s. The real fix was to move the matmuls off
the CPU.

### 3. Expert matmuls on the GPU, straight from the host cache ("host bank")

**What it does.** The GB10 is an integrated GPU that can use host pointers for registered memory. The cache
regions are now `cudaHostRegister`ed and wrapped in a `CUDA_HostBank` buffer type that the CUDA backend accepts
as weights. The small-batch `MUL_MAT_ID` runs on the GPU and reads the experts in place. Only the slot lookup
stays on the CPU, so the hook still fills the misses first.

**Why it works on the Spark when patch 0002 did not.** Patch 0002 let the GPU compute on the mapped file pages
directly. That measured about 19 s per token, because on the GB10 every GPU touch of a file page goes through the
UVM fault handler. Registered anonymous memory is mapped into the GPU before the kernel runs, so it never faults.
A later nsys profile showed the GPU reading the 4 KiB-page bank at about 208 GB/s, the same efficiency as its own
buffers. A hugepage bank would therefore not help the GPU side.

**Two bugs on the way.**

- **Misaligned blocks.** The CUDA kernels address an expert as `slot * nb[2] / type_size` blocks. The 12 layers
  with IQ2_XXS parts (66-byte blocks) have record sizes that are not a multiple of 66, so the first build read
  misaligned blocks and answered `????`. The fix pads the slot stride to a multiple of every part's block size.
- **32-bit offsets.** The MMQ and MMVQ kernels computed expert offsets in 32 bits. The Spark's bank fits under
  that limit, but the GH200's bank of 49k slots overflowed it.

**Effect.** **1.43-1.47 tok/s**, up from 0.84. The gap between fills fell from 12 to 4 ms per layer, and prefill
of 102 tokens took 18 s. The token was bound by the NVMe again.

### 4. io_uring reads

**Read shape.** A microbenchmark of the decode read pattern ([`tools/readbench.cpp`](../tools/readbench.cpp))
showed what matters on the PM9E1:

- The size of a read does not matter, because the block layer splits everything at 128 KiB.
- What matters is having the whole fill queued at once.
- 16 threads issuing 1 MiB `pread`s filled 7 records in 4.5 ms. One io_uring ring with the whole fill submitted
  did it in 3.9 ms.

**Measuring the drive.** fio over the store's real blocks gave 10.5-11.0 GB/s by any path. Blocks that were never
written read at 14.9 GB/s because the drive does not touch flash for them, so raw-device tests have to be limited
to the store's extents.

**Effect.** Reads 9.6 → 10.5-11.1 GB/s, decode **1.37 → 1.47 tok/s**, bit-identical.

**Two fixes in the same pass.**

- **I/O pool race.** A worker of the previous run could run a job of the next run.
- **THP on the host cache regions.** Transparent huge pages made 4 of 7 server starts on the GB10 produce NaN
  from the first prefill. THP is now opt-in there (`LLAMA_EXPERT_STORE_THP=1`); the cause is still unknown.

### 5. Asynchronous host-bank hook, slot maps on the GPU

This change is the base for everything after it. The slot-map lookup moves to the GPU. The CUDA backend posts
each layer's selected expert ids to a mailbox in mapped host memory and waits on a flag with
`cuStreamWaitValue32`. A poller thread on the CPU runs the fill and sets the flag. The host never blocks, so
decode graphs are captured as CUDA graphs again.

**Effect on its own:** neutral, because launch overhead was not the bottleneck. What it enables is reading
experts while the GPU is busy with something else, which is what every following step uses.

### 6. Self-prefetch

**What it does.** Right after the router, each layer posts its real expert ids without waiting. The GPU then
computes the shared expert and the latent down projection while the poller reads the misses. Only after that
does the real lookup wait. The order of the arithmetic does not change; only the read timing does.

**Effect.** **1.49 → 1.60 tok/s** (+7.7 % on the medians). The poller reads about 6.7 records per layer in about
4 ms while the GPU computes, and the layer went from about 7.7 to 7.0 ms.

### 7. Speculative prefetch v2 and its predictor inputs

An nsys profile of 24 decode tokens split the 6.67 ms layer into three parts:

| part of the layer | time | what happens |
|---|---|---|
| shared expert | 0.65 ms | the self-prefetch reads start |
| fill wait | about 3.4 ms | the GPU waits for the misses |
| expert matmuls, routed_up, next layer's attention and router | about 2.8 ms | **the drive is idle** |

At 11 GB/s those 2.8 ms are worth about 4.8 records, against 6.7 misses per layer.

**What it does.** During that idle window the engine reads the experts it predicts the **next** layer will miss.
The prediction applies the next layer's router to an approximation of the next layer's input.

**Why version 1 lost.** The first version was 1.33-1.35 vs 1.39-1.43 tok/s. It read the predictions through the
poller at full depth, so the next real request queued behind them and both competed for a saturated drive.

**Version 2.** It is built around never delaying a real read:

- a side reader thread with its own io_uring ring;
- at most 8 pieces of 1 MiB in flight;
- at most 16 records per post, best router scores first;
- **preemption by every real fill:** records already started are finished and adopted if the real fill wants
  them, records not started are handed back.

Only the poller thread touches the LRU, and per-slot states keep in-flight slots out of eviction. The real lookup
still plans the real ids, so the output cannot change.

**Predictor inputs.** The predictor improved in three steps (`LLAMA_EXPERT_STORE_PREFETCH_INPUT`):

| input | what the next router sees | result |
|---|---|---|
| router input (`PREFETCH=1` alone) | this layer's router input | 1.69-1.76 tok/s |
| **input 1** | the un-normed FFN input plus the shared expert's output, under the next layer's FFN norm | **1.60 → 1.80 tok/s** (+12 %), self-prefetch fill 6.7 → 4.1 records, about 70 % of started reads used |
| **input 2** | the stream plus the shared expert, put through the next layer's own residual mix (see below) | top-16 precision 0.614 → 0.675, **1.81 → 1.85 tok/s** (+2 %), fill 4.1 → 3.8 records |
| **input 3** | input 2 plus the resident experts' share of this layer's routed output, available from the split GEMV (step 10) | about +1 % on top |

Input 2 follows from how Kimi K3 is built: every 12th layer banks its residual stream, and every FFN input is a
learned mix of the banked checkpoints and the stream. The input-1 predictor ignored that mix, and an offline dump
of predictor against router (`tools/k3-pfdump` in patch 0003, scored with [`tools/pfeval.py`](../tools/pfeval.py))
showed precision of 0.34 on the checkpoint layers and 0.01 on layer 13.

Window depths of 4 and 16, and demoting unused predictions, all measured worse than the deployed setting
(depth 8, no demote).

### 8. Stall fixes: memory lock and `--cache-ram 0`

These change no tok/s figure. They remove rare multi-second stalls that showed up as "0.1 tok/s" stretches.

**The cause.** After a load the Spark has about 7 GB of headroom, and about 150 MB of the server was already in
swap. Every swapped page comes back as a major fault queued behind the expert reads, about 10 ms each. A few
hundred in a row make a stall of several seconds.

**Two fixes.**

- **Memory lock.** The engine locks its own memory, everything except the model file mappings, at the first fill:
  about 63 GB in 5-17 s, once (`LLAMA_EXPERT_STORE_MLOCK`, default on; the container needs `CAP_IPC_LOCK`).
- **No prompt cache.** llama-server's prompt cache is turned off. Each saved prompt of this model costs about
  1.3 GiB, and 8 different prompts had swapped 5.4 GB out.

**Effect.** Neutral on tok/s (1.82 vs 1.82), and `VmSwap` stays at 0.

### 9. Cache snapshot

**What it does.** At exit and every 5 minutes, the engine writes which experts are resident, in recency order
with their use counts (`LLAMA_EXPERT_STORE_SNAPSHOT`, about 6 bytes per slot). The next start re-reads those
records from the store before serving.

**Effect.** All 7980 slots, 52.5 GB, restored in 5.5 s. The hit rate over the first 84k lookups was 85 % instead
of 48 % from a cold start, and the first run after a restart reached 1.59 tok/s instead of about 1.3. Output is
unchanged by construction.

### 10. Split expert GEMV

**What it does.** Before this change the GPU waited for all misses (about 2.8 ms) and only then ran the expert
matmuls, although about 12 of the 16 selected experts were already in RAM. With `LLAMA_EXPERT_STORE_SPLIT=1` the
graph looks the experts up twice:

1. **Early lookup.** It returns right after the plan. Experts already in RAM get their slots, and misses point at
   a zero record. A fill thread starts the misses' reads.
2. **Work under the fill.** The GPU runs the shared expert, the matmuls over the resident experts and the next
   layer's predictor.
3. **Real lookup.** It waits for the fill. An 8-row matmul then computes the misses, and `ggml_set_rows` copies
   each miss's row into its position.

Every row is computed once, from the same slot, by the same kernel, and the weighted sum keeps its order, so the
output is bit-identical.

**Why the late matmul has only 8 rows.** The zero record's dummy rows cost real bandwidth, about 28 µs each,
because the GPU reads the host bank uncached. A 16-row late matmul gave back a third of the gain.

**Effect.** Medians of clean runs went from **1.851 to 1.914 tok/s** (+3.4 %), and to **1.935** with predictor
input 3. On average 11.6 of 16 experts are complete at the early lookup, and the GPU's wait at the real lookup
fell from about 2.8 ms to 1.2-1.4 ms. The gold md5 held on all 21 runs. This is the deployed configuration:
**1.93 tok/s**.

### 11. NVMe IOMMU passthrough

DGX OS boots with `iommu.passthrough=0`, so every `O_DIRECT` piece is mapped and unmapped through the SMMU.
Passthrough (`iommu.passthrough=1` in a GRUB drop-in, then a reboot) measured 1.87 vs 1.85 tok/s, which is within
noise. It was kept because it costs nothing. It is a node setting, not part of the engine.

### 12. Memory pressure valve

**Why.** The cache size (`--moe-cache-mib`) had only ever been checked with short prompts. A long prompt grows the
CUDA compute buffers by several GB. With no swap and a locked bank, the kernel's answer is the OOM killer, which
killed the GH200 pod on a 30k-token prompt.

**What it does.** The valve samples `MemAvailable` every 500 ms, and also on a PSI trigger where the process runs
as root. Below `LLAMA_EXPERT_STORE_RESERVE_MIB` (4096) it gives the least recently used slots back to the kernel
(`munlock` and `MADV_DONTNEED`). It re-arms them after 10 s of calm, and it never releases more than half of the
cache. A record is dropped whole or not at all, so the change is lossless.

**Effect on the Spark.**

| test | result |
|---|---|
| idle box | neutral: nothing released, MemAvailable about 8.6 GB |
| a 7 GiB process started during the runs | 5.6 GB released, 1.87-1.94 tok/s under pressure, gold md5 on every answer, all re-armed 10 s after the process exited |
| a 31k-token prompt | completed with no OOM; MemAvailable low-water 7.2 GB, zero slots released |

### 13. Second-chance eviction

**Why.** The Spark's cache holds about 10 % of the experts, so the eviction policy matters more than on the
GH200. The real top-16 selections of 700 decode tokens were replayed at the Spark's 7980 slots with
[`tools/cachepolicy.py`](../tools/cachepolicy.py):

| policy | hit rate, code / prose |
|---|---|
| plain LRU | 57.5 / 61.7 % |
| LRU plus a per-slot second-chance count of up to 3 | 59.6 / 64.1 % |
| S3-FIFO | 59.5 / 64.7 % |
| ARC | below the second chance |
| LFU with decay | below the second chance |
| Belady, which needs the future | 76.6 / 79.2 % |

S3-FIFO tied with the second chance. The second chance costs one byte per slot, so it went in
(`LLAMA_EXPERT_STORE_SECOND_CHANCE=3`).

**Effect, Spark A/B.** Interleaved A B A B legs, 30 runs, all with the gold md5:

- **1.93 → 1.96 tok/s (+1.8 %).**
- The engine's reported hit rate went from 76.3 to 77.1 %. That figure counts records the speculative reader
  brought in; the replay attributes roughly 15 points of it to speculation.
- The self-prefetch fill went from 3.9 to 3.8 records.

Second chance is in the published patch set but is **not yet in the deployed config**.

**Pinning measured in the same pass.** Pinning the I/O and poller threads to the GB10's big X925 cores
(`LLAMA_EXPERT_STORE_CPUS`) gave nothing, because they then compete with the eight matmul threads.

### Progression

Spark decode, warm cache, temperature 0. Each row's figure comes from that step's own A/B (see the note at the
start of this section).

| step | decode | per token |
|---|---|---|
| unpatched llama.cpp, mmap'd GGUF | 0.007 tok/s | 147 s |
| 0001 `MADV_WILLNEED` prefetch | 0.22 tok/s | 4.45 s |
| 0003 expert store, CPU matmuls, 16 threads | 0.71 tok/s | 1.40 s |
| `--threads 8` | 0.84 tok/s | 1.19 s |
| expert matmuls on the GPU from the host bank | 1.43-1.47 tok/s | 0.68-0.70 s |
| io_uring reads (A/B baseline 1.37) | 1.47 tok/s | 0.68 s |
| self-prefetch (A/B baseline 1.49) | 1.60 tok/s | 0.63 s |
| speculative prefetch v2, predictor input 1 | 1.80 tok/s | 0.56 s |
| predictor input 2 (A/B baseline 1.81) | 1.85 tok/s | 0.54 s |
| mlock + `--cache-ram 0`, snapshot, IOMMU passthrough | about 1.85 (no stalls, warm after restart) | 0.54 s |
| split expert GEMV + predictor input 3 (deployed) | **1.93 tok/s** | 0.52 s |
| memory pressure valve | 1.93-1.95 (neutral) | 0.52 s |
| second-chance eviction (measured, not deployed yet) | **1.96 tok/s** | 0.51 s |

Prefill on the Spark still goes through the copy path: every expert a ubatch uses is read from the store into
staging. A 102-token prompt took 18 s in the GPU-bank build. A 31k-token prompt prefilled at about 9.5 tok/s,
56 minutes in total. Long-context decode speed has not been measured cleanly.

## Where a Spark token's time goes now, and the hard limits

At 1.93 tok/s a token is about 0.52 s, or about 5.6 ms per MoE layer. Per layer:

- **Reads under the shared expert and the resident matmuls.** The self-prefetch fill of about 3.8-3.9 missed
  records, at about 0.72 ms per record (the drive's ceiling), runs under the shared expert, the early expert
  matmuls and the predictor. The GPU still waits about **1.2-1.4 ms** at the real lookup.
- **About 2.8 ms of GPU work with the drive busy on speculative reads.** This is the misses' matmuls, the merge,
  routed_up and the next layer's attention. That work is almost entirely weight streaming:
  - the next layer's Q6_K projections, about 370 MB at **about 210 GB/s, about 77 % of the LPDDR5X's 273 GB/s
    peak**;
  - the expert GEMVs from the host bank at about 208 GB/s;
  - gaps between kernels of only 0.07 ms per layer.
- **Once per token:** the output head (4.7 ms) and the dense FFN of layer 0 (3 × 0.93 ms).

What bounds further gains:

- **Misses.** The only part of a layer that fewer NVMe misses can remove is the 1.2-1.4 ms wait. Removing every
  miss would give about **2.46 tok/s**. That is the ceiling for anything that only removes reads: a bigger cache,
  a better policy, a perfect predictor, or remote RAM.
- **The drive.** The fills already run at the PM9E1's random-read ceiling of about 11 GB/s over written blocks.
  Lossless compression of the records was measured at 0-3.4 %, because IQ1_S is close to incompressible.
- **The GPU side.** There is no structural slack. Kernel-level gains are bounded by the roughly 23 % between
  210 GB/s and the memory peak.
- **Memory.** There is nothing left to give to the cache: the prompt cache is already off. In the replay, 20 %
  more slots would buy about as much as the best eviction policy.

## What did not work

| idea | result | why |
|---|---|---|
| Patch 0002: the GPU computes on the mmap'd experts in place | about 19 s/token | GB10 GPU reads of file pages go through UVM faults. Built in, off. |
| `MADV_COLD`, `vm.watermark_scale_factor=500` (mmap path) | no change | reclaim stayed on the critical path |
| Experts of some layers on LAN nodes via llama.cpp RPC | 0.30 tok/s with 0001; with the store, 0.68-0.70 vs 0.71 without RPC | the LAN is 40-90× slower than the NVMe, and round trips cost as much as the bytes they saved |
| Spark plus two Intel NUCs over a Thunderbolt mesh (2026-10 study, [`tools/distributed/`](../tools/distributed/)) | predicted 0-1 % bit-identical; 3-6 % with other kernels | The mesh does not reach the Spark, which has only a 2.5 GbE port (about 290 MB/s). A NUC RAM tier can carry only predictions 4+ layers ahead. Remote compute round trips (p50 1.9 / p90 3.3 ms) do not fit the ~2.5 ms window. The Arc iGPU runs IQ1_S no faster than the CPU (~23 GB/s). The NPU path needs Q4 requantization. Rejected. |
| Speculative decoding (draft or n-gram) | n-gram 7.9 → 8.05 tok/s on the GH200, not bit-identical | while NVMe-bound, each draft token selects its own experts |
| vLLM, SGLang, ktransformers | cannot run K3 on these boxes | ktransformers is x86 AMX only |
| Predictive prefetch v1 (on the poller, full depth) | 1.33-1.35 vs 1.39-1.43 tok/s | wasted reads compete with real ones on a saturated drive; v2 fixed the overlap |
| Prefetch window depth 4 / 16, demoting unused predictions | no better than depth 8 | depth 8 already fills the window |
| 16-row late matmul in the split GEMV | 1.91 vs about 1.93 tok/s | dummy rows read uncached host memory |
| THP on the host cache regions | NaN from the first prefill in 4 of 7 starts on the GB10 | cause unknown; opt-in only |
| Hugepage destinations for `O_DIRECT` | 3.9 vs 4.5 ms per fill in the microbenchmark | blocked by the THP problem above |
| Lossless compression of records | 0-3.4 % | IQ1_S grid indices are about 8.0 bits per byte |
| CPU pinning of the I/O threads to the X925 cores | no gain, slightly slower | contention with the matmul threads |
| Eviction policies other than second chance (ARC, LFU with decay, CLOCK) | at or below second chance in the replay | second chance was the simplest of the winners |
| Hopper TMA Q6_K GEMV on the GB10 | 5-10 % per kernel (about 1.5-2 % of a token) | not bit-identical on the GB10, whose stock kernel uses its own warp layout; opt-in (`GGML_CUDA_MMVQ_Q6K_TMA=2`) |

Items that only apply to the GH200 are measured there and do not transfer to the Spark:

- **io_uring loses on the GH200's virtio disk** (8.4 vs 9.6 tok/s), because one submitter feeds one blk-mq
  queue. The GH200 uses 32 `pread` threads instead.
- **Speculative prefetch loses at 99 % residency** (13.84 → 11.3-11.5 tok/s). Wrong guesses evict resident
  experts, and the predictor's GPU time exceeds the 0.14 ms per layer of real reads it could hide.
- **Neutral there:** `GGML_CUDA_GRAPH_OPT=1`, a spin-wait in the I/O pool, and Hopper variants of the GB10 MMVQ
  tuning.

## Open ideas

None of these has been measured end to end yet. Estimates come from the ledger.

- **Deploy second chance** (+1.8 %, measured).
- **Run the misses' matmuls as their records land.** A per-id wait inside `mul_mat_vec_q` would hide all but the
  last record's share of the misses' matmuls. Estimate: about 0.2 ms per layer (about 3 %), at the price of a
  kernel change.
- **A zero record the GPU caches.** A zero record in device memory, mapped into the bank's virtual range with
  CUDA VMM, would make the split GEMV's dummy rows free.
- **A better predictor.** Input 2's precision is 0.675. An oracle predictor would leave 2.39 records in the
  self-prefetch fill instead of about 3.8. The remaining error is the next layer's attention, which is known only
  after it runs.
- **A disjoint CPU split.** I/O and poller threads on a couple of X925 cores, matmul threads on the rest.
- **A GB10 variant of the Q6_K TMA kernel** that mirrors the GB10's stock warp layout, so it stays bit-identical.
- **The THP NaN on the GB10.** Fixing it would bring back hugepage `O_DIRECT` destinations.
- **A faster network link to the Spark.** With 10 GbE, a NUC RAM tier is estimated at +3-5 % bit-identical.
- **Prefill numerics that do not depend on the cache layout.** The suspect is MMQ stream-k over the slot-ordered
  expert list. Fixing it would make prefill changes testable bit for bit.

## The GH200 port

The same engine runs on a GH200 (Grace with 431 GiB of LPDDR5X, an H100 with 96 GB of HBM3 exposed as a CPU-less
NUMA node, NVLink-C2C between them, and a virtio NVMe at about 9.5 GB/s). Here the cache holds most of the
experts: 424000 MiB in host RAM plus a 24 GiB HBM tier, with about 98-99 % of decode lookups hitting. The
bottleneck becomes the GPU reading host memory over C2C plus the Q6_K spine.

Steady decode is **about 13.2-13.8 tok/s** on the 800-token coding run, with 2 × 256k context.

Features that differ from the Spark:

| feature | what it does on the GH200 | effect |
|---|---|---|
| Slot maps on the CPU (`LLAMA_EXPERT_STORE_GPU_SLOTS=0`) | faster than the async GPU path here | 13.65 vs 12.8 tok/s |
| No speculative prefetch, no split GEMV | the split GEMV needs GPU slot maps and no HBM tier | not used |
| HBM tier (`LLAMA_EXPERT_STORE_HBM_MIB`) | the most used experts are swapped into slots on the HBM NUMA node, which the GPU reads at 2-3 TB/s instead of about 345 GB/s over C2C | 7.77 → 9.58 tok/s at the time |
| Prefill from the bank (`LLAMA_EXPERT_STORE_MAX_BATCH=2048`, ubatch 2048) | only the misses are read; MMQ walks only the experts a batch uses | prefill of 512 / 1024 / 2048 tokens 23 / 20 / 16 → 54 / 90 / 126 tok/s |
| Prefill lookahead (`LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=1`) | the next layer's misses are read while the GPU computes the current layer | 16k-token prompt 98-106 → 127-145 tok/s, now read-bound at the drive's 9.6 GB/s |
| Hopper Q6_K TMA GEMV (`mmvq.cu`, on by default on sm_90) | streams weight tiles through shared memory with the bulk-copy engine and keeps the stock arithmetic | bit-identical; steady decode 12.99 → 13.25 tok/s (+2 %) |
| Reads | 32 `pread` threads in 1 MiB pieces (`LLAMA_EXPERT_STORE_URING=0`, `_THREADS=32`) instead of io_uring | one 6.6 MB record 2.9 → 1.2 ms |

Several GH200 fixes are in the shared code:

- 64-bit expert offsets in the MMQ and MMVQ kernels;
- `mbind` of the cache to the CPU nodes, so a populate cannot spill into HBM;
- the asynchronous mailbox carries the slot map by value, which fixed a crash with two sequences in one batch;
- the valve drops only whole 64 KiB pages inside a run of released slots.

The ledger records the full sequence of passes.

## Building and running

### Image

The CI in [`.github/workflows/images.yml`](../.github/workflows/images.yml) builds
`ghcr.io/xerootg/llamacpp-spark:<LLAMA_SHA, 12 characters>-g<short sha of this repo>`. The image targets arm64
with CUDA 13, sm_90 + sm_121 and `-DGGML_RPC=ON`. To build it locally from this repo:

```sh
. ./versions.env
docker buildx build --platform linux/arm64 -f docker/llamacpp-spark.Dockerfile \
  --build-arg LLAMA_SHA="$LLAMA_SHA" -t llamacpp-spark:local --load .
```

The Dockerfile checks out upstream llama.cpp at `LLAMA_SHA` (pinned in [`versions.env`](../versions.env)) and runs
`git apply` on `patches/*.patch` in order.

### From the fork

The `kimi-k3` branch of `xerootg/llama.cpp` has the same tree, one commit per patch. Build it like any llama.cpp
CUDA build. The flags the image uses are in [`docker/llamacpp-spark.Dockerfile`](../docker/llamacpp-spark.Dockerfile):

```sh
git clone -b kimi-k3 https://github.com/xerootg/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121" \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9-a+dotprod+fp16+sve+sve2+i8mm+bf16
cmake --build build -j
```

Use architecture `90` for a GH200. Install `liburing-dev` before configuring: without it the engine falls back to
the `pread` pool.

### The expert store

Build the store once from the GGUF. Put it on a local NVMe, ideally the same drive the server reads from, and do
not run the repack while a benchmark is running:

```sh
python3 tools/repack-experts.py \
  --gguf /models/unsloth/Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf \
  --out /store/kimi-k3-ud-iq1_s.experts --verify 200
```

The script uses only the standard library. It writes with `O_DIRECT`, at about 0.9 GB/s on the Spark (about
10 minutes for 545 GB), and checks random records against the GGUF. Keep the GGUF: the spine is still loaded from
it. `--index-only` prints the plan without writing anything.

### Server flags and environment (Spark)

These are the values of the Spark deployment. The Kubernetes manifests for both boxes live in a private GitOps
repo (`configs/kimi-k3/llama-server.yaml` for the Spark, `configs/kimi-k3-gh200/llama-server.yaml` for the GH200),
so the relevant settings are reproduced here.

```sh
llama-server \
  -m /models/.../Kimi-K3-UD-IQ1_S-00001-of-00014.gguf --alias kimi-k3 \
  --ctx-size 32768 --parallel 2 --batch-size 2048 --ubatch-size 512 \
  --n-gpu-layers 999 --n-cpu-moe 93 \
  --moe-cache-mib 52000 --cache-ram 0 \
  --threads 8 --threads-batch 20 --flash-attn on --jinja \
  --fit off --load-mode mmap --no-warmup -lv 4
```

What the less obvious flags do:

| flag | why |
|---|---|
| `--n-cpu-moe 93` | places the routed experts in host memory, where the store serves them |
| `--moe-cache-mib 52000` | the RAM budget of the cache: 128 GB minus the 45.5 GB spine, KV and compute buffers, the 6.7 GB prefill staging and headroom |
| `--threads 8` | keeps the matmul threads off the little cores |
| `--load-mode mmap` | must be forced on the GB10 (see above) |
| `--fit off` | the fit pass ran the box out of memory, because "device memory" is the whole box |
| `--cache-ram 0` | turns off the prompt cache (see step 8) |
| `-lv 4` | shows the store's log lines, for example `host bank` and `expert store: locked ... MiB`, the hit/miss summaries and the split-GEMV stats |

Environment:

| variable | Spark value | meaning |
|---|---|---|
| `LLAMA_EXPERT_STORE` | `/store/kimi-k3-ud-iq1_s.experts` | enables the engine |
| `LLAMA_EXPERT_STORE_SNAPSHOT` | `/snapshot/kimi-k3-ud-iq1_s.k3sn` | cache snapshot; needs a writable directory |
| `LLAMA_EXPERT_STORE_THREADS` | `16` | the `pread` pool, used when io_uring is off |
| `LLAMA_EXPERT_STORE_SELF_PREFETCH` | `1` | step 6 |
| `LLAMA_EXPERT_STORE_PREFETCH` | `1` | speculative prefetch v2, step 7 |
| `LLAMA_EXPERT_STORE_PREFETCH_INPUT` | `3` | predictor input, step 7 |
| `LLAMA_EXPERT_STORE_SPLIT` | `1` | split expert GEMV, step 10 |
| `LLAMA_EXPERT_STORE_RESERVE_MIB` | `4096` | the valve's reserve, step 12 |
| `LLAMA_EXPERT_STORE_SECOND_CHANCE` | `3` (recommended, not yet deployed) | step 13 |

Defaults the Spark relies on:

- io_uring reads (`LLAMA_EXPERT_STORE_URING`, on when built with liburing);
- 1 MiB read pieces (`LLAMA_EXPERT_STORE_CHUNK_KIB=1024`);
- the GPU host bank and the GPU slot maps;
- `LLAMA_EXPERT_STORE_MLOCK=1`;
- `LLAMA_EXPERT_STORE_VALVE=1`;
- `LLAMA_EXPERT_STORE_MAX_BATCH=32`, so prefill uses the copy path.

The GH200 adds:

- `LLAMA_EXPERT_STORE_URING=0` and `LLAMA_EXPERT_STORE_THREADS=32`;
- `LLAMA_EXPERT_STORE_GPU_SLOTS=0`;
- `LLAMA_EXPERT_STORE_MAX_BATCH=2048` with `--ubatch-size 2048`;
- `LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=1`;
- `LLAMA_EXPERT_STORE_HBM_MIB=24576` and `LLAMA_EXPERT_STORE_HBM_DECAY=512`;
- `LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=6144`;
- `--moe-cache-mib 424000`, `--ctx-size 524288`, `--threads 16`, `--threads-batch 32`.

Container requirements:

- `CAP_IPC_LOCK`, or a large enough `RLIMIT_MEMLOCK`, for the memory lock. Without it the engine warns and runs
  unlocked.
- A termination grace period long enough for the exit snapshot write. The deployment uses 120 s.

### Checking a build

Run the fixed request sequence (one warm-up, then the same prompts) at temperature 0 against a known-good build,
with `cache_prompt: false`, and compare md5s. Without a Spark, use the small-MoE check described in the
[README](../README.md#validating-a-patch-without-the-spark).

## Where the numbers come from

- **[`docs/experiments.md`](experiments.md):** every experiment with its setup, raw runs and verdict, including
  the dead ends.
- **[`patches/README.md`](../patches/README.md):** what each part of each patch does, and every knob.
- **[`docs/uvm-experiment.md`](uvm-experiment.md):** the GH200 unified-memory bandwidth study behind the HBM
  tier.
- **The deployment notes** (per-pass results for each box) live with the private manifests. The figures quoted
  here are copied from them into the ledger.
