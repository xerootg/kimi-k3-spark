# GLM-5.3 current state

As of 2026-10-31. This page separates deployed behavior, measured experiments and estimates. The [experiment ledger](experiments.md) is the chronological record; older roadmap priorities are superseded by the results below.

## What runs today

The Spark serves full GLM-5.3 UD-Q2_K_XL, chosen as the smallest quant above the quality cliff of the largest model; Flash was rejected. Local K3 pods were scaled down and their files retained. The GH200 still serves K3; moving it to GLM remains the owner's decision. Sources: [roadmap and decision log](roadmap.md), [GLM bring-up ledger](experiments.md#glm-53-weights-laid-out-store-v2-ssd-tier-g53x-serveidx-2026-10-29).

| Deployed setting or measured step | Value | Source |
|---|---|---|
| Short-prompt decode progression | 3.76 → 3.85 tok/s with cache expansion → 4.25 with byte strides → 4.90–4.93 with prediction | [Ledger: cache, byte-stride and predictive-prefetch legs](experiments.md) |
| Expert cache / context / request slots | 84000 MiB adopted, pending rollout (80000 MiB deployed) / 65536 tokens / one slot | [Production profile configuration](glm-floor-profile-2026-10-30.md), [GH200 service trade](gh200-glm-plan.md) |
| Cache expert slots | 6859 restored experts; distinct layout groups constrain capacity | [Prefill analysis](experimental/0007-prefill-analysis.md), [placement assumptions](../tools/glm53/PLACEMENT.md) |
| Spark store and active knobs / eviction policy | Store v1; split expert GEMV, byte strides, predictive prefetch input 3, second chance, snapshot restore and memory valve. **LFU adopted after window l: +5.49 % (5.40525 → 5.70225 tok/s), bit-identical in all eight timed passes**; final hit rate 92.0 → 92.7 %, decode reads 234.1 → 212.6 GB. Patch 0018 v2 with headroom and no-growth fixes goes into the stack; manifest sets `LLAMA_EXPERT_STORE_POLICY=lfu`, `LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144` (owner commits) | [Ledger: window l](experiments.md#glm-53-the-candidate-windows-second-pass-2026-10-30-evening), [profile configuration](glm-floor-profile-2026-10-30.md), [patches](../patches/README.md) |
| Production short-prompt decode | Cache **84000 MiB pending rollout**, expected **~5.8 tok/s warm** after window cap (+2.7 %, bit-identical). Current deployed check: **~5.6 tok/s warm with LFU**; image `ge4e4290`, argo-things `8a2d14b`. Public check after window i: r0/r1 5.114/5.584 tok/s, md5 `9449ae5738` both; volatile LFU state warms after restart | [Live check](../evidence/prodcheck2.txt), [ledger](experiments.md#glm-53-the-candidate-windows-second-pass-2026-10-30-evening) |
| Greedy decode identity gate | md5 prefix `9449ae5738`; full digest `9449ae5738f2e56f99ab7ab6e9626eb5` for the standard fixture | [Ledger](experiments.md), [prefill analysis](experimental/0007-prefill-analysis.md) |
| Prefill with larger ubatches | 512: 24.0 tok/s; 1024: 37.4; 2048: 65.9; production now uses 2048. **0010 v3 measured +45 % prefill** (2048-token prompt 72.5 → 106.3, 8192-token 70.7 → 102.0 tok/s), decode and bytes unchanged on pb2; **adoption pending clean-tree confirmation and image** | [Third-pass ledger](experiments.md#glm-53-the-candidate-windows-third-pass-2026-10-31-night) |
| Long-prompt qualification | 30000-token measured prompt: 493.8 s, 60.76 tok/s; decode after long prompts: about 4.0–4.2 tok/s | [Prefill analysis](experimental/0007-prefill-analysis.md), [ledger](experiments.md) |

The larger ubatch amortizes repeated weight passes. The detailed prefill analysis measured the larger-ubatch legs with the experimental miss-only patch; it explicitly left the unpatched large-ubatch comparison open. The later ledger records the production setting and rejection of that patch, but does not supply a separate unpatched large-ubatch timing table. Treat the headline as the reported rollout result with that provenance caveat, rather than proof that every production prompt sustains it. Sources: [analysis](experimental/0007-prefill-analysis.md), [latest ledger](experiments.md).

## Where a token's time goes, and the remaining levers

The unprofiled budget is about 204 ms/token. These four large classes come from instrumented node intervals, not an exact partition of uninstrumented wall time. The profiler reduced throughput from 4.90 to 4.75 tok/s, and its class totals must not be corrected by subtracting a uniform overhead. Source: [per-token profile](glm-floor-profile-2026-10-30.md). Window m (clean tree, LFU off, 64-token nsys captures): **Three buckets of similar size: dense-spine GEMV ~68 ms (patch 0027 targets ~7), host-bank expert GEMV ~56 ms at ~132 GB/s (scheduling plan: 10-19 ms ceiling), read waits ~40 ms (hedge/latency work); the 124 ms / 59 GB/s first reading was a classification error, corrected the same night.** Source: [window m breakdown](experiments.md#glm-53-the-candidate-windows-second-pass-2026-10-30-evening).

| Large class | ms/token | Meaning | Source |
|---|---:|---|---|
| Spine device matmuls | 76.6 | Attention projections, shared/dense FFNs, router/indexer and head; semantic subgroups are mixed | [Profile](glm-floor-profile-2026-10-30.md) |
| Routed expert host-bank matmuls | 72.5 | Split execution visits 17 positions per layer for 8 real experts and 9 zero/scratch positions | [Profile](glm-floor-profile-2026-10-30.md) |
| Host hooks / slot-map handoffs | 42.0 | Includes about 28.5 ms of remaining exposed lookup wait; not pure CPU overhead | [Profile](glm-floor-profile-2026-10-30.md) |
| Visible attention primitives | 3.4 | Flash attention, sorts and RoPE; projections are in the spine class | [Profile](glm-floor-profile-2026-10-30.md) |

Other small operations and the unprinted tail account for the remainder. The old unprofiled “GPU/host floor” estimate is superseded: removing all observed lookup wait alone would leave about 175.6 ms, or 5.69 tok/s, conditional on everything else staying fixed. Neither that bound nor logical expert bytes measures physical DRAM traffic. Sources: [profile](glm-floor-profile-2026-10-30.md), [weight inventory](glm53-kernels.md).

The order below follows the measured floor analysis, with the NUC integration and prefill experiments called out separately. Targets are screening estimates, overlap each other and must not be added.

| Rank / lever | Status | Expected gain and next gate | Source |
|---|---|---|---|
| A: Skip dummy expert rows | **Adopted in production 2026-10-30 (argo-things 07f7100): combined with D +13.1 % (4.80 → 5.43 tok/s), bit-identical**; patch 0009, flag `LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1`; image ge0ba4b3 with both knobs on in the manifest | Combined window u passed all eight md5 gates; individual gain was +4.8 % | [Ledger: combined window u](experiments.md#glm-53-the-candidate-windows-second-pass-2026-10-30-evening) |
| B: GB10 Q5_K/Q6_K spine GEMV | Parked / proposed | Target 4.46–8.92 ms from improving those groups by 10–20%; Hopper results are precedent, not a GB10 result | [Profile](glm-floor-profile-2026-10-30.md) |
| C: Real IQ host-bank kernels | Parked / proposed after dummy isolation | Target 7.25–14.49 ms for a 10–20% interval improvement; overlaps dummy-row savings | [Profile](glm-floor-profile-2026-10-30.md) |
| D: Hook/mailbox work and CUDA graph coverage | **Adopted in production 2026-10-30 (argo-things 07f7100): combined with A +13.1 % (4.80 → 5.43 tok/s), bit-identical**; patch 0012, flag `LLAMA_EXPERT_STORE_HOOK_LEAN=1`; image ge0ba4b3 with both knobs on in the manifest. CUDA graphs: 126 of 129 reused per request, not a lever. Last off/on real-hook totals 15757.192/17463.834 ms; idle_sleeps 124406/349994 | Combined window u passed; individual hook-lean gain was +5.8 %; next: real-lookup wait split | [Ledger: combined window u](experiments.md#glm-53-the-candidate-windows-second-pass-2026-10-30-evening) |
| E: Short-context DSA fast path | Built as patch 0011 (`LLAMA_DSA_SHORT_CTX_SKIP=1`); **aborts on the first decode token** (`GGML_ASSERT(buffer)`, an unallocated index tensor); fix in flight. The long-context variants 0015/0016 measured neutral on an 8192-token prompt | Target 1–5 ms overall; sort-only ceiling about 1.01 ms | [Ledger: candidate windows](experiments.md#glm-53-the-candidate-windows-first-pass-2026-10-30-afternoon) |
| F: Better miss readiness / resident NUC service | First real measurement (window i, fixed 0013 against nuc2): **−7 % decode; not adopted, redesign pending** (early-hook post, RAM-only set). NUC pod scaled back to 0 | Off 5.388–5.398 vs on 4.969–5.120 tok/s at 2500 µs, md5 gold; 29.5 % fallback. The 1500 µs leg served 0 rows and failed the offload gate | [Runner output](../evidence/runner4-i.txt), [ledger](experiments.md#glm-53-the-candidate-windows-second-pass-2026-10-30-evening) |
| Prefill ubatch increase | Measured; deployed setting reported | 24.0 → 65.9 tok/s at ubatch 2048; see provenance caveat above | [Ledger](experiments.md), [analysis](experimental/0007-prefill-analysis.md) |
| Further prefill batching / direct bank path | **Measured 2026-10-30: host-bank prefill (patch 0010) 72 → 102 tok/s on a 2k prompt, 71 → 99 on 8k, long-prompt decode bytes identical**; not adopted yet: it carves its landing slots out of the decode cache (−8 % decode) and the short standard request decodes different bytes; both fixes in flight | Gate: standard md5 `9449ae5738`, decode within 1 % of baseline, prefill ≥ 95 tok/s | [Ledger: candidate windows](experiments.md#glm-53-the-candidate-windows-first-pass-2026-10-30-afternoon) |
| MTP decode multiplier | Parked after measurement; the exact-verify patch 0014 is being rebased onto the prefill-bank tree | No positive gain established; as-is result below supersedes optimistic study scenarios | [MTP ledger](experiments.md#glm-53-mtp-speculative-decoding-is-a-net-loss-today-and-changes-output-2026-10-30), [study](experimental/mtp/STUDY.md) |

## NUC tier: components proven, production integration still open

CUDA-exact means matching the CUDA implementation's float bits, including its quantization, reduction order, MUFU activation tables, FTZ and NaN behavior. The owner verified CPU reference and laptop iGPU stages against CUDA on 96 cases across representative layers; served computed rows matched too. Zero-row NaN sentinels deliberately request local fallback. The optimized CPU implementation preserves that scalar oracle's arithmetic. Sources: [exact result](../tools/glm53/EXACT-RESULT.md), [CPU result](../tools/glm53/cpu/SOL-RESULT-2.md), [kernel documentation](glm53-kernels.md).

The roadmap reports a fast-exact iGPU candidate (`G53X_EXACT=2`) at about 1.5× fast-path cost and CPU math around 2 ms/expert. Its standalone `FASTEXACT-RESULT.md` is absent here, as is `NUC-DEPLOY.md`; the checked-in [exact selector](../tools/glm53/g53x_exact.inc) maps any nonzero setting to the same exact kernels. Thus the fast-exact report is owner-reported progress, not a verified distinct deployed mode in this checkout. Sources: [roadmap](roadmap.md), [remote audit](../evidence/remote-tier-result.md). Optimized CPU synthetic timings reach roughly 1.5–2.3 ms/expert with 12 workers, excluding sockets/disk and using a closed-form activation subset; they are not NUC end-to-end measurements. Source: [CPU result](../tools/glm53/cpu/SOL-RESULT-2.md).

| Service / hop | Measured latency | Relation to overlap budget | Source |
|---|---|---|---|
| Spark → nuc2 empty RTT | p50 0.104 ms, p99 0.206 ms | Network itself is small | [Survey findings](../survey/FINDINGS.md) |
| Spark → nuc2 resident fast, k=1 | p50 0.917 ms, p99 2.023 ms | Small resident requests look feasible; fast arithmetic is not exact | [Survey findings](../survey/FINDINGS.md) |
| Same, k=8 | p50 3.493 ms, p99 4.613 ms | Already exceeds mean layer spacing | [Survey findings](../survey/FINDINGS.md) |
| Spark → nuc2 resident scalar exact, k=1 | p50 11.448 ms, p99 12.720 ms | Does not fit | [Survey findings](../survey/FINDINGS.md) |
| Spark → nuc2 cold prototype fast, k=1 | p50 7.913 ms | Synchronous read/repack/compute does not fit | [Survey findings](../survey/FINDINGS.md) |
| Later laptop disk tier, fast, k=1 / k=8, all cold | p50 3.8 / 15.6 ms | Improved implementation, different host; still no proof of timely NUC cold rows | [Disk result](../tools/glm53/DISK-RESULT.md) |

The planning window is about 2.72 ms: the mean spacing at 4.9 tok/s across 75 MoE layers. It is not a measured post-to-needed deadline. Start with a small resident request and measure the lower-tail overlap window; the engine audit proposes a 2500 µs deadline and one row, then a two-row A/B. The row cap is global across hosts. Sources: [survey](../survey/FINDINGS.md), [remote audit](../evidence/remote-tier-result.md).

The disk tier is implemented: preallocated host-USM LRU, in-place reader-thread repack, resident-first compute and per-row NaN deadline fallback. Resident, cold and LRU-hit rows passed byte comparisons against the corresponding kernel reference. The request allocation counters were zero in both exact and fast merged paths. Zero faults is a separate unmet gate: the merged laptop runs had 5 minor faults in exact mode and 1 in fast mode, all in i915 GEM mappings; major faults were zero. Recheck on the NUC driver. Sources: [disk result](../tools/glm53/DISK-RESULT.md), [merge result](../tools/glm53/MERGE-RESULT.md).

What remains is a real routing trace and held-out replay for ownership, exact-mode NUC latency/tails, the engine's remote-tier patch under hardware test, and the service image. Placement code currently demonstrates synthetic ownership only; its deployment script selects non-exact fast mode and cannot certify production. The audit has offline parser, deadline, NaN-hole and fallback regressions, but no CUDA build or end-to-end remote equivalence/throughput run. The image build/deployment is not established by the supplied records. Sources: [placement result](../tools/glm53/PLACEMENT-RESULT.md), [placement procedure](../tools/glm53/PLACEMENT.md), [remote audit](../evidence/remote-tier-result.md).

Use unique owner-filtered supported sets, not both disk servers' all-record advertisements. The current wire protocol sends actual inputs and has no predictor weight-prefetch channel, so placing an expert on remote SSD does not make its cold record arrive early. Local fallback must remain byte-exact and available. Sources: [placement](../tools/glm53/PLACEMENT.md), [remote audit](../evidence/remote-tier-result.md).

## Dead ends measured today

| Experiment | Result and one-line reason | Source |
|---|---|---|
| Store v2 on Spark, patch 0006 | About 4.82 vs 4.90 tok/s, −1.7%: already-aligned GLM records gain no read advantage and padding adds 4% bytes; retain reader compatibility for NUC stores, keep Spark v1 | [Ledger](experiments.md), [patch README](../patches/README.md) |
| Miss-only prefill, experimental 0007 | About 21 vs 24 tok/s at ubatch 512 despite 39% fewer logical read bytes: serialized hit copies, warming and synchronization cost more than saved reads | [Ledger](experiments.md), [analysis](experimental/0007-prefill-analysis.md), [design](experimental/0007-prefill-design.md) |
| MTP as-is | About 90% draft acceptance yet 12–16% slower: 4.49 baseline vs 3.97 MTP and 3.76–3.87 exact-columns MTP, with different output hashes; batch verification loses prediction and changes arithmetic beyond expert matmuls | [Measured ledger](experiments.md), [study](../evidence/mtp-study.md), [exact-column scope](experimental/mtp/PATCH-0008.md) |

The exact-column harness does pass singleton-equivalent expert arithmetic with its knob enabled; that is not whole-model equivalence. Attention, indexer, dense dispatch/fusions and batch prediction still need work. The archived MTP study's speed scenarios predate the measured loss. Sources: [ledger](experiments.md), [MTP archive](experimental/mtp/README.md), [study](experimental/mtp/STUDY.md).

## GH200: decision pending the owner

The choice is whether to reserve a reversible GLM UD-Q5_K_XL benchmark window and eventually displace K3. The plan estimates a 562.46 GB GGUF, 534.35 GB active expert store and roughly 75% expert residency with its proposed tier budgets. Estimated warm decode is roughly 4–9 tok/s with about 2 tok/s downside under less favorable routing; none is measured GLM GH200 throughput. Capacity, hit rate, indexer/KV memory and long-prefill peaks need hardware gates. Source: [GH200 plan](gh200-glm-plan.md).

The service cost is losing the existing K3 backend's two 256k slots. Spark GLM currently offers one 64k slot and a different model/quant; it cannot transparently absorb those sessions. The plan recommends a benchmark window, not immediate production replacement. Its old Spark prefill comparison predates today's larger-ubatch result and should not be reused as current evidence. Keep K3 artifacts/config for rollback; no GH200 conversion is recorded. Sources: [GH200 plan](gh200-glm-plan.md), [decision log](roadmap.md), [latest prefill ledger](experiments.md).

## Method and ownership

Patches are opt-in knobs, not automatic speed claims. The patch series moves from mmap prefetch and the rejected mmap zero-copy path to the explicit expert store, byte-addressed cache slots, graph prediction and store-format compatibility. Experimental prefill and MTP patches remain archived outside the Dockerfile's applied patch directory. The broader byte-stride type audit establishes alignment coverage, not a new measured GLM gain. Sources: [patches 0001–0006](../patches/README.md), [byte-stride audit](experimental/0004b-byte-stride-types.md), [prefill ledger](experiments.md).

Workers (sol and Copilot) draft bounded changes; the owner reviews them and the hardware gates decide. Every candidate is A/B'd from the same immutable cache snapshot, with the original greedy fixture and a byte-identical md5 gate before accepting a speed claim. A changed store needs its own equivalent warm snapshot: the first store-format comparison was invalid because its candidate started cold. Keep model, prompt, context, cache budget and output fixture matched; report any deliberate difference. Sources: [decision log](roadmap.md), [patch README](../patches/README.md), [store-format ledger](experiments.md), [profile measurement procedure](glm-floor-profile-2026-10-30.md).

Exact math, allocation/fault gates and end-to-end speed are separate gates. A remote identity run must actually use remote rows, rather than merely pass through local fallback; absent, refused or late rows must fall back locally. No production offload speedup, image rollout or GH200 migration is implied by a worker report or synthetic replay. Sources: [remote audit](../evidence/remote-tier-result.md), [merge result](../tools/glm53/MERGE-RESULT.md), [placement result](../tools/glm53/PLACEMENT-RESULT.md).

## energy

[`tools/tokwatt.py`](../tools/tokwatt.py) uses `PROM_URL` (or `--prom-url`)
to query `homeassistant_sensor_power_w{entity="sensor.server_cabinet_power"}`
at 10 s resolution. Supply `--start EPOCH --end EPOCH --tokens N`, or
`--legs-log FILE` containing `LEG name start=<epoch> end=<epoch> tokens=N`.
Epoch seconds may be fractional; tokens must be actual decoded output counts.
`python3 tools/tokwatt.py --demo` runs a synthetic self-test without network.

For duration T and output N: throughput = N/T; cabinet watts W = mean of
finite nonnegative samples in [start,end); baseline B = median in
[start−120,start); incremental watts ΔW = W−B. Tokens per incremental joule
= N/(T×ΔW) = (tokens/s)/ΔW, also tokens/s per watt. Whole-cabinet tokens/kWh
= N×3,600,000/(T×W). Energy is estimated from the sampled mean over the full
window, not an integral across missing intervals. Nonpositive ΔW gives NA
for incremental efficiency. Missing samples are omitted, counts and partial
coverage flags are printed, and an empty interval gives NA for its metrics.
Partial-window figures assume the remaining samples are representative.
The query rejects samples older than 20 s to limit Prometheus lookback filling;
repeated Home Assistant readings can still hide an upstream sensor outage.

Optional owner leg pattern: run `TOKWATT=1 bash evidence/glm-legs-v2b.sh`
on the Spark and retain stdout. The opt-in addition prints one LEG marker per
measured request (r1/r2), using `timings.predicted_n`, `predicted_ms`, and local
response receipt time. Warmup r0 is omitted. Start is estimated as receipt minus
decode duration, excluding prefill approximately; Python startup and response
transport shift both boundaries. For precise windows, capture absolute decode
start/end in the server with a synchronized clock. Relative server timestamps
alone cannot place historical runs on the Prometheus timeline. Preserve LEG
lines when using spark-windows.sh wrappers: their tail/grep filters can drop them.

The Fort Collins cabinet contains roughly ten machines, including the Spark.
Reported ~468 W idle-ish and 538 W peak are context, not fixed baselines or
new measurements. Other machines, UPS losses and baseline drift affect the
meter; compare otherwise quiet, matched runs. The preceding two minutes can
include warmup or another decode (especially back-to-back legs), biasing the
baseline upward. Arrange a quiet two-minute interval before each measured
window for clean attribution. The 10 s sensor cadence limits short decode
accuracy; longer windows and clock synchronization help. Incremental watts
understate a dedicated box's draw because idle power is subtracted; do not
present incremental efficiency as the Spark's total wall-power efficiency.
