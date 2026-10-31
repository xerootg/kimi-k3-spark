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

- Status: **done 2026-10-29** (see the ledger).
- Result:
  - The cost was nuc2's iGPU sleeping (RC6) between requests, not marshaling.
  - Measured candidates: forcewake alone 2.12 tok/s; the CPU half (AVX-VNNI on the shared arena) 2.103 tok/s at a 0.66 ms round trip; the Spark-side fix +0.5 %.
  - Next: an interleaved A/B of forcewake + memwarm against the CPU half, then deploy the winner plus the S2 engine (`gc4efdca`).

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

### 2b. Spark: cheap power-state and I/O-path tests (added 2026-10-29)

Same lesson as the nuc2 RC6 find and the `cpu_dma_latency` win: power management costs that nobody measured.
Each is an interleaved A/B with gold md5; none changes arithmetic.

| test | evidence | expected | status |
|---|---|---|---|
| Lock the GB10 SM clock at its 3003 MHz max (`nvidia-smi -lgc`) | decode runs at a steady 2411 MHz (application clock 2418), no throttle reason active | 0-3 %; the dense kernels stream at 77 % of memory bandwidth, part of the gap may be issue rate | queued with the pool agent |
| `--threads` re-sweep (4, 6, 8, 10) | 8 dates from the CPU-matmul era; the CPU does far less now | 0-2 % | queued with the pool agent |
| Multi-user throughput with the pool on (1, 2, 3 clients) | dense weights are read once per batch; last multi-client number predates the pool | a throughput headline, not single-user speed | queued with the pool agent |
| Polled NVMe completions (`nvme.poll_queues` + `IORING_SETUP_IOPOLL`) | ~4 reads of ~0.7 ms gate each layer, each completion is an interrupt | 50-100 us per layer, 1-2 %; needs a reboot and a small reader change | after the aligned store |
| Reclaim the 6.7 GB prefill staging during decode | lazy staging thrashed on 10-19, before mlock and the valve existed; 6.7 GB = +13 % cache slots | about +1.5 % (replay: +20 % slots = best eviction policy) | after the aligned store |
| NVMe APST off (`nvme_core.default_ps_max_latency_us=0`) | APST on with a 100 ms cap; the drive is never idle long during decode | probably nothing; free on the same reboot | with the polled-I/O reboot |

### 2c. A faster link to the Spark (hardware, user's call)

- **Cheap, ~$100:** a USB 3.2 10GbE adapter on a Spark USB-C port (20 Gbps USB 3.2, not USB4). Latency through USB is worse, so it suits a NUC RAM byte tier (raw records, predicted layers ahead): the study put that at +3-5 % with 10G against 0-1 % at 2.5 GbE.
- **Proper, ~$250:** the ConnectX-7's QSFP28 cage with a 10GBASE-T SFP+ module in a QSA adapter, direct to a Thunderbolt 10GbE adapter on nuc2. Lowers every expert round trip too.

### 2d. Engine hot-path audit findings (2026-10-29, `docs/audit-2026-10-29.md`)

Ranked by expected gain per hour. All bit-identical by construction.

| # | change | expected | status |
|---|---|---|---|
| 1 | skip zero-record ids inside `mul_mat_vec_q` (about 9 dummy ids per layer at 28 us) and set SPLIT_LATE=16 | +5-6 % | done as patch 0009 (GLM-5.3 +4.8 %, in production). Q's K3 variant (op-param sentinel) measured 2.052 -> 2.094 tok/s (+2.0 %, gold, sync excess reads 0.16 -> 0) on 2026-10-29, not merged; `docs/experimental/q-audit-items.md` |
| 2 | bound the speculative reader's bytes in flight (own piece size, stop issuing on preempt) | +3-6 % (guess) | built by Q (`PREFETCH_CHUNK_KIB`, `PREFETCH_PAUSE`), not measured: the Spark went to GLM-5.3 mid-window; parked in `docs/experimental/q-audit-items-on-c4efdca.patch`, needs a rebase onto the stack |
| 3 | fused SiTU GLU op with the gate/up MUL_MAT_ID fusion (27 tiny kernels per layer) | +2-3 % | after Q |
| 4 | remote tier: REMOTE_ROWS 3-4 and a shorter deadline now that the round trip is 0.66 ms | +1-2 % | pool agent, env only |
| 5 | store v2 needs a per-entry file stride (5 offset sites), not only a header change; `max_sectors_kb` is already 512 | part of the aligned store | with step 2 |
| 6 | fill and spec thread wake-ups on the critical path (condvar per layer) | +0.5-1 % | after Q |
| 7 | io_uring SINGLE_ISSUER, DEFER_TASKRUN, COOP_TASKRUN | 0-1 % | built by Q (`URING_DEFER=1`, Spark kernel 6.17 / liburing 2.5 qualify), not measured; parked with item 2 |
| 8 | predictions should not promote hits or earn second chances | +/-1 %, replay first | replayed by Q (`tools/cachepolicy.py`): about 1 % fewer misses on both K3 traces, below A/B resolution; built (`PREFETCH_PROMOTE=0`), not measured; parked with item 2 |
| 9 | byte strides in mmvq, drop the IQ2_XXS slot padding (+3 % cache) | +0.3 % | later |
| 10 | k3cpu tail: re-execute a claimed-but-unfinished chunk; SO_BUSY_POLL | +1 % on the remote leg | pool agent |

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

### Coming hardware: a fleet of A10s (24 GB each)

Enough cards to hold the whole 594 GB model in VRAM (25+ cards). That makes K3 a multi-GPU, multi-node problem instead of a
disk-streaming one: no SSD wall, and the 45.5 GB spine split across GPUs. The new limits are each card's ~600 GB/s, PCIe 4.0
(no NVLink on the A10) and the network between hosts.

Planning, to do before the cards arrive:
- **Placement:** layers or experts per GPU, and pipeline vs expert-parallel vs tensor-parallel for the spine.
- **Host specs to learn:** CPU memory channels, PCIe topology, NICs (RDMA?).
- **Bit-exactness on sm_86:** whether the GB10/GH200 bit-exact discipline carries over to Ampere kernels.

The fork `xerootg/llama.cpp` is a research fork and will not be upstreamed. Its default branch is `kimi-k3`.

## GLM-5.3

As of **2026-10-31, through the third pass (night)**. Sources: [experiment ledger](experiments.md#glm-53-the-candidate-windows-third-pass-2026-10-31-night),
[GLM status](GLM-STATUS.md), [NUC decode verdict](remote-tier-2026-10-31.md),
[NUC prefill study](nuc-prefill-offload-2026-10-31.md),
[host-bank GEMV plan](experimental/host-bank-gemv-plan.md) and [THP plan](experimental/thp-bank-plan.md).

**Spark production: 5.20 cold / 5.75 warm tok/s, cache 84000 MiB with LFU, gold md5 `9449ae5738`**,
argo-things `60335ee` (owner's live check, superseding the ledger/status pages' pending-rollout wording).
Patches 0009 (split dummy-row skip), 0012 (hook lean) and 0018 v2 (LFU) remain on;
predictor input 3 and prefetch top-8 stay. Combined 0009/0012 measured +13.1 %, LFU +5.49 %,
and the capacity window +2.7 %; these are separate A/Bs, not additive gains.

**Measured candidates and dispositions:**

| candidate | measured result / decision |
|---|---|
| cap: 80000 → 84000 MiB with LFU | CAP80 5.637/5.697, CAP84 5.793/5.858, CAP80b 5.630/5.713 tok/s; observed timed passes gold, +2.7 %. Hit rate 92.9 → 93.6 %; lowest MemAvailable 8.74 GB (CAP84), 8.68 GB (CAP84b), against 4.29 GB reserve. CAP84b timings absent from named evidence. Adopted. |
| x: 0010 prefill bank v3 | On pb2, without 0009/0012/0018: 2k prefill 72.5 → 106.3, 8k 70.7 → 102.0 tok/s, about +45 %. Standard decode ~4.9 tok/s unchanged; long decode 4.03 → 4.02 / 4.10 → 4.07, bytes unchanged. Shared landing replaces staging; decode cache budget unchanged. Accepted pending clean-tree x3 and image. Supersedes v1's −8 % decode and changed short-request bytes. |
| o2: 0021 prefill order independence | At 16k / ubatch 2048, stock and patched each gave the same md5 `546cda4032` across three cold passes, prefill 68.4–68.7 tok/s. No layout variation reproduced on either clean tree; model-free stock FAIL / patched PASS remains the demonstrated distinction. Candidate for a prefill-testability guarantee, not urgent adoption; runner log lost, per-pass results retained. |
| i2 / i4: NUC decode hedge | i2 −2.8 %, gold: reader-backend override regressed io_uring to pread. Fixed in 0013c; i4 still ~−1 %, gold, at ~1.8 ms RTT. Pending-row wait and posting/collector overhead outweigh wins against the local tail; parked, off for throughput. |
| NUC prefill offload | Not viable with today's kernels. Decode GEMVs do not reuse weights across tokens; exact batched arithmetic and a Spark prefill-MMQ oracle must precede any integration. Conditional study scenarios are not measured gains. |
| af: 0023 affinity | +0.2 %, neutral; FIFO slightly worse; gold throughout. Kept off. |
| pc: 0029 prefetch top-k 6/7 vs 8 | Neutral within noise despite fewer physical bytes; gold. Keep top-8. |
| dp: 0025 demand readers; sh: 0022 shared-expert reorder | Both neutral; kept off. Shared expert already overlaps lookup wait with input 3. |
| 0011 v2 DSA skip; 0019 v2 fast exit; 0005b bounded speculative-read wait | Neutral: 0011 v2 is lossless and stays off (v3 staged); 0019 neutral to small and stays off; 0005b kept without a knob. |
| 0015/0016 long-context attention; 0017 predictor inputs 4/5 | Attention variants neutral, off; inputs 4/5 slower, keep input 3. |
| 0012c wider prediction | −1.5 %, gold, useful predictions ~67 → 41 %; dead for adoption. Wait-split instrumentation retained off by default. |
| hb: 0024 hostalloc bank | Neutral at 72000 MiB (~5.1 tok/s), gold; disables memory valve. Dead for adoption despite standalone bank-read bandwidth; write-combined leg skipped as unsafe. |
| MTP | As-is −11 % and changed bytes despite high acceptance; dead for current adoption. Exact patch did not apply in j; y failed to link, so no new timed exact result. |
| gv / gm / bg: GEMV schedules | Dense standalone bench: 1232 matches, zero mismatches; synthetic inventory 66.04 → 59.16 ms/token (~6.9 ms potential). gm and bg ran only gold OFF controls before faulty init-log gates aborted; ON schedules unmeasured, gates fixed and reruns queued. |
| thp: 0026 bank-only huge pages | Inconclusive: sampled-record checker fails even OFF controls after gold responses. Checker fix and rerun queued; no demonstrated bank corruption or gain from this window. |

**Open questions:** the corrected nsys short-context budget ranks the next decode levers:
**(1) dense-spine GEMV ~68 ms/token, (2) host-bank expert GEMV ~56 ms/token at ~132 GB/s,
(3) read waits ~40 ms/token**. Dense schedules have ~6.9 ms/token standalone potential;
bank schedules have a **10–19 ms/token ideal ceiling**, not a measured engine gain.
The old 124 ms / 59 GB/s bank attribution mixed dense and routed MMVQ and is superseded.
These captured-token durations (clean tree, LFU off) are not an exact partition of unprofiled production
wall time; short capture includes prefill. Host API time overlaps GPU work and is not an additive savings budget.

Record latency has a **1.1 ms isolated floor**, rising to **1.5–3.5 ms in the engine** under
**~129 records/token** of mostly speculative traffic; about half the early-issued records are incomplete
at the real hook. Top-k reduction measured neutral, so fewer bytes alone has not solved queueing.
The [remote audit](remote-tier-2026-10-31.md) corrects early i4 counter summaries: exposed remote wait
is occupancy while local reads may also be pending, not incremental delay to subtract from wall time.
Joint local/remote critical-path timing is missing; no precise positive selective-hedge expectation is established.
NUC decode stays parked unless RTT/selection economics change; NUC prefill needs new exact batched kernels.

The ctx sweep found no context cliff: decode ~4.2 tok/s from 2k through 32k; the earlier cached-prompt
stall was not reproduced and remains unexplained. Earlier trees showed layout-dependent prefill;
o2's stable clean-tree results do not establish that every dispatch/layout is invariant.
Prefill v3 must repeat on the committed stack before production adoption.
THP must demonstrate retained huge backing, exact bytes, safe pressure/regrow and a net win after
padding's capacity cost; the [plan](experimental/thp-bank-plan.md) does not prove a NaN root cause.

The **GH200 still serves K3**. The [decision memo](gh200-decision-2026-10-30.md) options stand:
uninterrupted K3, a reversible K3 byte-stride window, or a GLM UD-Q5_K_XL benchmark.
Byte strides offer **+3.2 % nominal host capacity only**, not measured speed; guard, restore,
pressure and identity gates remain. GLM transfer/capacity/outage budgets and the loss of K3's two 256k
slots still require an owner decision; the memo's GLM speed estimates remain unmeasured.

**Queued windows and the decisions they must settle** (not measured gains):

| window | work | decision / gate |
|---|---|---|
| gm | 0027 dense GEMV schedules | Fixed init-log gate; measure ON versus OFF on the engine with gold bytes. ~6.9 ms/token standalone potential is not an engine result. |
| bg | 0028 bank GEMV schedules | Fixed init-log gate; test real/dummy rows, exact arithmetic and engine speed. 10–19 ms/token is an ideal bandwidth ceiling; preserve accumulation order. |
| thp | 0026 bank-only huge pages | Fix false-positive checker; require retained huge backing, record/output identity, safe valve pressure/regrow and net speed after capacity loss. Keep staging and hostalloc separate. |
| z | 0020 late GEMV ready flags | Compile fix queued after undefined `__assertfail`; prove overlap, gold bytes and safe composed reader/GPU teardown. Still unmeasured. |
| x3 | Prefill bank v3 on the clean committed tree | Confirm pb2's +45 % prefill, unchanged decode/bytes and bounded landing reservations; then decide production image adoption. |
| i5 (later, conditional) | Age-limited NUC hedge | Only if RTT/selection economics change; unstarted/young-read selection is unproved. Require a repeatable positive gold A/B with local fallback intact. |

Source review and standalone benches do not certify engine gains. Use matched frozen snapshots,
standard greedy md5 gates and measured A/Bs before adoption.

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
- 2026-10-29: hot-path audit done (docs/audit-2026-10-29.md); items 1, 2, 7, 8 go to agent Q, 4 and 10 to the pool agent.
- 2026-10-29: plan review added the power-state/I/O tests (2b), the link options (2c) and a read-only code audit of the engine's hot path for performance left on the table.
- 2026-10-28: the fork's default branch is `kimi-k3` (research fork, not for upstream); patch set e844a2a published as tag `kimi-k3-e844a2a`.
- 2026-10-29: the local pool switches from K3 to GLM-5.3 UD-Q2_K_XL (user: "the smallest version of the largest model", above the 2-bit cliff; Flash rejected as a smaller, shallower model). K3 files stay on disk; the GH200 keeps serving K3.
- 2026-10-29: shards live on the NAS (slow but reliable); every pool member gets a full store copy so expert ownership is a table, not a copy.
- 2026-10-29: bounded, testable work goes to GitHub Copilot (coding agent for repo-only tasks, `copilot-agent` CLI for tasks that need the laptop iGPU); Claude reviews and reruns every gate before committing.
- 2026-10-30: adopt 0009 + 0012 together on Spark: image ge0ba4b3, argo-things 07f7100, 4.80 → 5.43 tok/s (+13.1 %), all eight timed passes md5 gold; predictor input 3 stays after inputs 4/5 lost.
- 2026-10-30: keep neutral 0005b; leave neutral 0015/0016 off. LFU v1 crashes, prefill bank v1 costs −8 % decode and changes short-request bytes, and upstream MTP costs −11 % and changes bytes; none is adopted.
- 2026-10-30: LFU 0018 v2 adopted after window l: +5.49 %, eight timed passes gold; production image ge4e4290, argo-things 8a2d14b, live check 5.114 cold / 5.584 warm tok/s, md5 gold. Neutral 0011 v2 and 0019 stay off; 0012c (−1.5 %) is not adopted.
- 2026-10-30: remote tier not adopted at decode: window i −7 % with about 30 % refusals/fallback. The audit corrects the posting account and bounds even ideal I/O hiding at about 6.7 tok/s. Park RAM-only / prepare-protocol redesign behind an unproved prefill/batch > 1 case; decode work moves to rl/cap/hb/gv, with no cheap predictor ensemble win.
- 2026-10-30: set 84000 MiB as the cache-capacity budget ceiling to test against 80000 in window cap; 88000 leaves too little projected reserve margin. Replay's 5.6 % fewer bytes/token is not a measured speedup or load-survival guarantee.
- 2026-10-30: 0021 adoption pending o2 at 16k/ubatch 2048. Window o is order-independent by construction, decode gold and neutral at 2k; confirm the failing shape before re-baselining long-prompt golds or gating prefill changes.
- 2026-10-30: queue o2/cap/rl/dp/hb/gv/sh/af/z/x/y/m; ctx is running. z is rebased, x uses prefill v3 with identity gates, and y follows x. GH200 remains K3 pending the owner's choice among uninterrupted service, byte-stride (+3.2 % nominal capacity, unmeasured speed), and a reversible GLM benchmark.
- 2026-10-31: adopt 84000 MiB with LFU after cap's +2.7 %, observed timed passes gold and ~4.4 GB above valve reserve. Production argo-things `60335ee`; live check 5.20 cold / 5.75 warm tok/s, md5 gold, supersedes pending-rollout notes.
- 2026-10-31: accept prefill bank v3 pending clean-tree x3 and image: pb2 measured about +45 % prefill with decode and bytes unchanged; shared landing replaces staging without taking decode cache.
- 2026-10-31: park NUC decode tier: hedge −2.8 % with reader-backend regression, then ~−1 % after the fix. At ~1.8 ms RTT, pending-row wait and overhead outweigh wins against the local tail; exposed-wait counters are not additive delay. Revisit age-limited i5 only if economics change. Today's kernels also make NUC prefill offload unviable.
- 2026-10-31: rank remaining levers dense-spine GEMV (~68 ms/token, gm ~6.9 ms potential), bank GEMV (~56 ms at ~132 GB/s, bg 10–19 ms ceiling), then read waits (~40 ms). Queue gm/bg/thp/z/x3; neutral affinity/top-k/demand readers/shared reorder stay off, 0012c/0024 hostalloc/MTP stay rejected. GH200 remains K3; byte-stride is +3.2 % capacity only.
