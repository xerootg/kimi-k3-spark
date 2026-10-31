# Next floor — 2026-10-31

Planning arithmetic, not new hardware measurements; decimal MB/GB throughout.
Sources: [ledger](experiments.md) (window v, record latency rl, nsys m, third pass), [closure](experimental/host-bank-gemv-plan.md#outcome-2026-10-31), [inventory](evidence/glm-windows-2026-10-30/legs-v.out) lines 6–12, [GH plan](gh200-glm-plan.md), [memo](gh200-decision-2026-10-30.md).
Warm Spark baseline is ~5.8 tok/s, T0=1000/5.8=172.414 ms; buckets D/B/W≈68/56/40 ms leave R=8.414 ms for this calibrated model.
The captures contain prompt work and overlapping host activity: R is a calibration remainder, not a measured fourth bucket. Hold context, routing, predictor and miss counts fixed below.

## 1. Change routed-expert quant only

Each gate/up/down part has N=6144×2048=12,582,912 weights, N/256=49,152 blocks; ordinary current record=49,152×(74+74+98)=12,091,392 B.
[Quant inventory](glm53-kernels.md#1-quant-inventory-http-range-reads-of-the-gguf-headers-only) distinguishes three dense blocks from 75 routed layers; the three dense blocks stay unchanged here.
The actual routed mix also has one IQ3_XXS gate/up exception and FOUR IQ4_XS down layers (136 B/block), not three dense layers standing in for those exceptions; see closure's kernel/bytes inventory.
Current routed payload P0=8×49,152×[2×(74×74+98)+71×98+4×136]=7,333,478,400 B/token; mean record=P0/(75×8)=12,222,464 B.
This gives active store=256×75×12,222,464=234,671,308,800 B≈235 GB (excludes MTP); ledger's 238.17 GB v1 includes MTP, and v2 padding raises it to 247.50 GB.
For uniform replacement of ALL routed parts: record E=3×49,152×b; token P=75×8×E; r=P/P0. These records are already 4096-aligned; no slot padding charged to GEMV.
Block sizes b=66/50/82/84 are in [repo type table](../tools/glm53/inventory.py); 50-byte IQ1_S is K3's dominant GH200 expert type, with mixed IQ2_XXS exceptions ([GH plan](gh200-glm-plan.md)).

| Routed quant | b B/256 weights | E B/record | P GB/token | r | D / B / W ms | T0−96×(1−r) ms | tok/s |
|---|---:|---:|---:|---:|---|---:|---:|
| Current UD mix | mixed | 12,091,392 typical | 7.3334784 | 1 | 68 / 56 / 40 | 172.41 | 5.80 |
| all IQ2_XXS | 66 | 9,732,096 | 5.8392576 | .796247 | 68 / 44.59 / 31.85 | 152.85 | 6.54 |
| all IQ1_S | 50 | 7,372,800 | 4.4236800 | .603217 | 68 / 33.78 / 24.13 | 134.32 | 7.44 |
| all IQ2_S | 82 | 12,091,392 | 7.2548352 | .989276 | 68 / 55.40 / 39.57 | 171.38 | 5.83 |
| all Q2_K | 84 | 12,386,304 | 7.4317824 | 1.013405 | 68 / 56.75 / 40.54 | 173.70 | 5.76 |

Formula: B'=56r, W'=40r, D'=68, speed=1000/(R+D'+B'+W'); ratio uses the full current mix rather than just its typical record.
Against ONLY the typical 12,091,392-B record, ratios would be 198/246=.804878, 150/246=.609756, 246/246=1, 252/246=1.024390.
Scaling W assumes unchanged ~1.35 unpredicted records/layer and byte-proportional exposed latency; fixed I/O overhead, changed cache capacity, routing and kernel efficiency can invalidate it.
Unsloth's published quant quality/perplexity table: **not in the repo**; the repo quotes model sizes and measured type inventories, not that table. No local quality measurement supports any alternative.
MD5 gates measure stability for a fixed quant/configuration, not quality; a changed quant should not be required to match today's text hash.
A new Spark perplexity window needs trusted source weights (avoid repeated lossy requantization), candidate GGUFs/stores, supported decode AND prefill kernels, fixed held-out text/tokenizer/context/stride, and a current-quant reference.
Prepare artifacts and a bank-aware loss evaluator first (ordinary full-GPU loading cannot fit); reserve exclusive RAM/GPU, record NLL/token and perplexity on identical tokens plus task samples, memory/throughput, then restore production/snapshot. No duration or quality tolerance is established.

## 2. Smaller dense spine, experts remain today's Q2 mix

Inventory gives Q5_K=7.586709504, Q6_K=1.063157760, Q8_0=5.064474624, Q4_K head=.535265280 GB/token: first three=13.714341888; including head=14.249607168.
At current ~200 GB/s, 13.714/.068≈202 GB/s. Ledger v measures Q5 7.5867/.038819=195.4 and Q8 5.0645/.024280=208.6 GB/s: 71.6–76.4% of 273 (ledger rounds 72–77%; v uses 270).
Q4_K blocks are 144/256 B versus Q5 176/256, Q6 210/256, Q8 34/32. All-Q4 lower-byte scenario: P'=7.586709504×144/176+1.063157760×144/210+5.064474624×144/272+.535265280=10.152787968 GB/token.
Saving=4.0968192 GB/token, or 4.0968192/.200=20.48 ms; D≈47.52 ms, B/W=56/40 unchanged, T≈151.93 ms→6.58 tok/s.
Q4_K_M is a mixed recipe, commonly retaining selected Q6 tensors; its exact GLM spine inventory is not in the repo. Retaining all current Q6 gives 10.486923264 GB/token, saving 3.762683904 GB/18.81 ms→6.51 tok/s.
These 18.8–20.5 ms scenarios require converting the large Q8 tensors too; they are not a guaranteed Q4_K_M range. Retaining additional sensitive tensors saves less; inspect an actual candidate header first.
Attention/indexer/shared-expert precision can affect quality and routing even when routed weights stay fixed; use the same held-out loss/task window. Stock scheduling already streams well; closure's neutral dense-kernel window supplies no additional speedup to add.

## 3. Two independent NVMe drives

[rl output](evidence/glm-windows-2026-10-30/legs-rl.txt) measures ~1.1 ms isolated (9–11 GB/s single-stream), ~2.7 ms with two background streams, not a 40-ms sequential bandwidth transfer.
Model loaded latency L(load)=1.1+queue(load). Halving per-drive load and assuming linear queue penalty: Lhalf=1.1+(2.7−1.1)/2=1.9 ms; W'=40×1.9/2.7=28.15 ms/token.
Thus saving≈11.85 ms, T≈160.56 ms→6.23 tok/s. Optimistic complete return to isolated floor: W'=40×1.1/2.7=16.30 ms, saving23.70, T≈148.71 ms→6.72 tok/s.
Expected planning budget is ~28 ms wait (~16 ms optimistic floor), not automatically 20 ms from doubling bandwidth. Readiness overlap/tails, load imbalance and shared PCIe can change either result.
Stripe the store across genuinely independent drives, or route layer-parity stores to separate reader queues; parity needs balance over the predictor's concurrent traffic, not merely equal layer counts.
A second 4 TB NVMe is overkill for this ~235 GB active store (~118 GB/drive split); a smaller compatible drive dedicated to store data suffices, allowing metadata, staging and free-space margin. No local price quote exists.
First settle it with matched warm one/two-drive runs, frozen snapshots, per-drive physical bytes and loaded record latency plus exposed wait; preserve the original store for rollback.

## 4. GH200: capacity changes the problem

The memo's rounded 96 GB HBM +441 GB host (ledger:431 GiB, unit difference) and 9.6 GB/s NVMe support Q2's entire ~235 GB active expert store in host RAM, with spine/KV/workspace in HBM and ample host reserve.
Once populated and verified fully resident, Q2 has W=0: no decode NVMe expert reads, regardless of predictor misses. Startup loading and other process I/O remain separate.
K3 reference: 13.65–13.84≈13.7 tok/s warm, mixed records up to 7.827456 MB, smaller6.451200-MB IQ1_S records on 80/92 layers; it is not a same-model benchmark ([memo](gh200-decision-2026-10-30.md), GH plan).
GH plan calibrates effective host-bank GEMV at ~340 GB/s, not the advertised fabric peak; Q2 bank≈7.3334784/340×1000=21.57 ms (versus Spark 56). K3's ~34 MB/82µs profile also includes HBM-tier effects, so this is provisional.
GH plan's K3-derived no-miss floor51–91 ms minus (9.8−7.3334784)/340×1000=7.25 ms suggests Q2≈43.75–83.75 ms, ~12–23 tok/s, ONLY assuming comparable spine/other costs.
That broad extrapolation brackets K3 13.7; it is not a measured GLM forecast. GLM attention/context and IQ kernels differ; measure its HBM spine and host GEMV before narrowing it. Spark 68 ms must not be carried unchanged into HBM.
Q5 active store534.345940992 GB exceeds practical host capacity; GH plan budgets400.312–428.423 GB resident (~75–80%), 16.698 GB/token experts and 19.716 GB/token non-experts.
Q5 bank≈16.698/340×1000=49.11 ms; its NVMe wait persists. Plan's no-miss floor70–110 ms plus 1000×16.698×(1−hit)/9.6 gives 35/87/174 ms I/O at 98/95/90% hits.
Hence memo's ~4–9 tok/s warm (~2 downside with uniform routing) remains appropriate pending measurement; Spark shows why loaded per-record latency and prediction waste can make bandwidth-only I/O optimistic. Residency fraction is not hit rate.

## 5. Owner ranking and deciding measurement

1. Prepare a GLM-Q2-in-RAM GH200 test first: largest structural gain without reducing quant precision. Stage/verify artifacts and sm_90 build while K3 stays up; take one ≤60-min interruption including rollback, with a hard stop leaving restoration time.
   One warm short/context-matched decode profile with zero expert reads, measured HBM/host allocations and bank bandwidth settles the speed claim. Retain K3 config/store/snapshot; verify K3 gold/service after restoration. Abort rather than extend if preparation/fit fails; no production replacement implied.
2. For keeping Spark, test expert IQ2_XXS quality first (~6.54 tok/s model); IQ1_S (~7.44) is more aggressive. One paired held-out perplexity/task window decides whether the byte savings are acceptable before kernel/store integration investment.
3. Test a Q4_K_M spine header and paired quality/loss window (~6.5 tok/s only under the explicit inventories above); attention/indexer sensitivity makes the exact tensor recipe decisive.
4. Buy/borrow a smaller second NVMe only after a two-drive loaded-latency/engine A/B shows ~28 ms exposed wait or better. Same quality, modest expected gain; capacity alone is no reason for another 4 TB.
5. Defer GH200 Q5 until Q2-in-RAM answers GLM suitability/performance; first deciding measurement is actual Q5 warm hit rate and loaded miss-wait distribution, not K3 speed or nominal aggregate RAM.
