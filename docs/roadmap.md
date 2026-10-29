# Kimi K3 roadmap: what we are doing next, in order, and where it stands

Living document. Update it whenever a step starts, finishes or changes, with the date and the measured numbers. The ledger
(`docs/experiments.md`) keeps the full record of every experiment. This file is the plan and its status.

## Goal and rules

- **Goal.** A research demonstration that pooled commodity hardware gives sovereign, usable inference of a very large model (Kimi K3,
  2.8T MoE). Reliability first, speed second.
- **Quality rule.** No quality loss: output bit-identical at temperature 0, relative to the UD-IQ1_S checkpoint.
  - **Spark gold md5:** warm-up `b2c79009…`, 200-token coding run `e801484d…`.
  - **GH200:** 800-token run `ac78af0e5d` with `cache_prompt: false` on an idle server.

## Where we are (2026-10-28 evening)

**Spark production: 2.048-2.053 tok/s, gold md5** (1.93 on 10-26, 1.98 this morning).

| change today | effect |
|---|---|
| second-chance eviction | about +1.8 % |
| `/dev/cpu_dma_latency` held at 0 on the Spark (argo-things `configs/kimi-k3/spark-dma-latency.yaml`) | +3.5 % |
| remote expert tier on nuc2 | neutral on speed |

**The remote tier.** nuc2 (ASUS NUC 15 Pro, Core Ultra 5 225H, Arc 130T, 64 GB) computes 6967 experts (46 GB) bit-exactly on its iGPU. It runs as the k8s pod `kimi-k3-k3x-nuc2`, and the engine uses at most 2 remote experts per layer. Output is identical with any box missing, and NVMe reads drop by 13-17 %. It does not pay yet: the engine's round trip is p50 ~2.0 ms against ~1.24 ms from a bare client, and the Spark's window is ~1.3 ms.

**Hard limits, see the ledger.**
- The disk is ~80 % busy at about 4.1 GB per token.
- Zero misses would give ~2.46 tok/s, because the spine is 45.5 GB read per token.
- Batching or speculation does not reduce misses per accepted token.

## Plan, in priority order

### 1. Remote tier: remove the marshaling (RUNNING since 2026-10-28)

Workflow `k3-remote-marshaling` (run `wf_2a8d1716-f97`), steps:
1. Measure: timestamp every remote request at both ends.
2. Build, in parallel:
   - a zero-copy nuc2 server: one host-USM arena shared by CPU and iGPU, fixed request slots, no H2D/D2H copies, fewer submissions, flag-based completion;
   - the Spark-side fix;
   - a bit-exact AVX-VNNI CPU kernel.
3. Integrate: split each request's experts between iGPU and CPU, then an interleaved Spark A/B with gold md5.
4. Adversarial verification.

Done when the engine's round trip sits inside the window and the tier is faster than tier-off in an interleaved A/B.

- Status: running.
- Result: _pending_.

### 2. Spark: aligned expert store (next on the Spark)

- **Evidence** (`tools/readbench.cpp`, 2026-10-28):

  | store | GB/s |
  |---|---|
  | written 10-18 | 11.07 |
  | fresh sequential copy | 11.9-12.0 |
  | fresh copy, reads starting on 512 KiB boundaries | **12.9-13.0 (+17 %)** |

- **Plan:**
  1. Store format version 2: a record stride padded to 512 KiB, with the engine still reading only `record_size`, so the bank does not grow. About +3 % file size.
  2. Engine support for the version-2 header.
  3. Repack from the fresh copy (`/srv/kimi-k3/store/kimi-k3-ud-iq1_s.experts.fresh`).
  4. Interleaved A/B against the old store.
  5. If it wins, swap the files and drop the old store.
  6. Then check whether the fresh-copy gain decays with age; if it does, add a periodic rewrite (rewrite cost ~2 min at 5.8 GB/s).
- **Expected:** about +8-10 % tok/s, the disk being the main wall.
- **Status:** not started; waits for the workflow's Spark windows to end.

### 3. Spark: the CPU computes missed experts as their records land

The GPU waits for all of a layer's misses, then runs one 8-row late matmul. Instead, the GB10's idle Arm cores compute each missed expert's row as soon as its record arrives, during the fill wait when the memory bus is mostly idle. The GPU then waits only for the last row.

- **Bit-exact:** an Arm (i8mm/SDOT) port of the bit-exact CPU kernel from step 1, with the same spec and MUFU tables.
- **Expected:** about +3-5 %.
- **Status:** not started; depends on step 1's CPU kernel.

### 4. Spark: GB10 variant of the Q6_K TMA GEMV

The Hopper kernel is 5-10 % faster per kernel but not bit-identical on the GB10. The variant mirrors the GB10's stock warp layout so that it stays bit-identical.

- **Expected:** about +1.5-2 % per token.
- **Status:** not started.

### 5. Speculative decoding: the only path past the 2.46 tok/s ceiling

- **Existing K3 speculators** on Hugging Face: RadixArk/Kimi-K3-DSpark (acceptance 4-5.5 per step on code and math) and lightseekorg/kimi-k3-eagle3.1-mla.
- **Steps:**
  1. Measure their acceptance against our IQ1_S target. DSpark needs K3's aux hidden states, `l_out` of layers 7/23/51/67/83. The 8 sample answers are already generated: scratchpad `spec/k3gen.json`.
  2. Write batch-invariant verify kernels, so a K-token verification computes each token exactly as single-token decode does. Without them, speculation is not lossless.
  3. Cost model with the remote tier absorbing the extra misses that rejected drafts cause.
- **Expected:** about 3-3.5 tok/s if acceptance holds and steps 1-2 land.
- **Status:** scoped; depends on steps 1 and 2.

### Not doing (measured or reasoned out)

| idea | why not |
|---|---|
| Output head on the Spark CPU | < 1 %; the bus is busy then |
| Thunderbolt 40G cable between the NUCs | the Spark is not on the mesh; Linux has no RDMA over Thunderbolt |
| Pipelining whole layers onto NUCs | ~1.6-1.9 tok/s; NUC memory bandwidth is too low for the dense part; not bit-identical |
| Corals | 8 MB SRAM, USB 3; the problem is bytes, not ops |
| Tensor parallelism over 2.5 GbE | ~4 gathers per layer cost more than the NUCs' share saves |

### Hardware options, decide after step 1

- **ConnectX-7 to a NUC with RDMA.** The Spark's two QSFP28 cages are idle. A ConnectX-4/5 Lx in a Thunderbolt enclosure on nuc2 would give RoCE with a few-µs round trip. Worth it only if, after step 1, the network is the largest remaining slice (today ~0.2 ms of ~2.0 ms).
- **More pool boxes.** nuc1 and nuc3 have a single memory channel (~45 GB/s); nuc (Minisforum i7-12650H, DDR4 dual channel ~51 GB/s, UHD iGPU) would mainly be a CPU-kernel box. The iGPU memory cap on 32 GB boxes is still unknown.

## Cluster layout for this work

| node | role |
|---|---|
| nuc2, nuc3 | reserved for Kimi: soft taint `kimi-k3/reserved=true:PreferNoSchedule`, label `kimi-k3/expert-node=true` |
| nuc | monitoring stack (moved off nuc3) |
| nuc1 | Jellyfin (QSV) |

## Decision log

- 2026-10-28: route 1 (remote expert tier) over route 2 (layer pipelining).
- 2026-10-28: Qwen3-Coder retired from nuc2 for good.
- 2026-10-28: nuc2/nuc3 reserved for Kimi.
- 2026-10-28: Spark priority order: aligned store, then CPU misses-as-they-land, then the GB10 Q6_K kernel.
- 2026-10-28: the remote tier stays on although it is speed-neutral (fewer NVMe reads, the pooling demonstration).
