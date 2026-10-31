# Host-bank GEMV: evidence correction and experiment plan
Local evidence audit updated after windows gm/bg; measurements below are supplied Spark outputs, not new sandbox GPU runs.
Sources: `evidence/csv/glm-{short,long}_cuda_{gpu_kern_sum,gpu_trace,api_sum}.csv`; derived audit: `evidence/nsys2-analysis.txt`.
## Correct denominator and family
`nsys_breakdown.py` classifies **all** `mul_mat_vec_q` as routed: 124.018/127.919 ms per short/long captured token is not host-bank time.
Grid Y=9 (early) or 8 (late), IQ types 17/18/23, isolates bank MMVQ: **55.5597/56.8651 ms**; device MMVQ accounts for **68.4584/71.0543 ms**.
Short contains prompt processing: 71×63 IQ3 down launches, 74×63+1 IQ2 late launches, 4×63+1 IQ4 downs; long has exactly 64 decode passes.
Thus 64 is a capture normalization, not 64 steady short decode intervals. Short bank decode ≈55.56×64/63=56.44 ms, slightly less after removing prompt MMVQs.
7.3334784 GB/55.5597 ms=132.0 GB/s (≈130 GB/s with 63-pass correction), **not 59 GB/s**. GPU busy141.40 and gaps51.96 span193.35 ms/captured token, not unprofiled173.91.
## Kernel inventory: short capture
Exact name key: `void mul_mat_vec_q<(ggml_type)T, (int)1, (bool)F, (bool)S, (bool)H>(...)`; F=fusion, S=small-K, H=halve-iters.
Each row below specifies the complete template key and grid X×Y×1; signature shared in kern_sum. Totals are capture ms; /64 column uses captured-token normalization.
| T,F,S,H; grid | role | instances; /64 | calls per MoE layer in steady decode | total ms; ms/64 | mean µs |
|---|---|---|---|---|---|
|17,0,0,0; 2048×9|IQ2_XS early gate AND up, unfused|9326;145.719|2 on74/75 layers|1456.619;22.7597|156.189|
|18,0,0,0; 2048×9|IQ3_XXS early gate AND up exception|126;1.969|2 on1/75|22.334;0.3490|177.252|
|18,0,0,0; 6144×9|IQ3_XXS early down|4473;69.891|1 on71/75|1185.135;18.5177|264.953|
|23,0,1,0; 1536×9|IQ4_XS early down|253;3.953|1 on4/75|64.569;1.0089|255.213|
|17,1,0,0; 2048×8|IQ2_XS late fused gate+up|4663;72.859|1 on74/75|343.879;5.3731|73.746|
|18,1,0,0; 2048×8|IQ3_XXS late fused gate+up|63;0.984|1 on1/75|8.447;0.1320|134.081|
|18,0,0,0; 6144×8|IQ3_XXS late down|4473;69.891|1 on71/75|453.384;7.0841|101.359|
|23,0,1,0; 1536×8|IQ4_XS late down|253;3.953|1 on4/75|21.456;0.3353|84.807|
Early=42.6352, late=12.9245 ms/64; steady total=225 early+150 late launches/token (5/layer). Shared/device fused Q5_K is NOT routed.
## Bytes and bandwidth: estimates, not hardware traffic counters
A tensor has 6144×2048 weights: IQ2_XS=74/256×weights=3.637248 MB; IQ3_XXS=98/256×weights=4.816896 MB; IQ4_XS=136/256×weights=6.684672 MB (decimal).
Payload=8×[2×(74×3.637248+1×4.816896)+71×4.816896+4×6.684672]=7333.4784 MB/token; record padding is not weight traffic.
Using the earlier profile's approximate 7 resident/1 missing experts (NOT measured in these CSVs), B_early=7×tensor_MB, B_late=1×tensor_MB; fused reads two tensors.
In table order, estimated MB/launch → GB/s = MB/(mean µs/1000): 25.461→163.0;33.718→190.2;33.718→127.3;46.793→183.3;7.274→98.6;9.634→71.9;4.817→47.5;6.685→78.8.
These are conditional useful-bandwidth estimates: maps, actual misses, addresses and physical reads are absent; no exact per-kernel bandwidth or per-layer identity can be recovered.
Late IQ3 down is slowest by that model; fused IQ3 exception next. Early gate/up are unfused, so do not blame gate/up fusion for the whole bucket.
Eight expert positions are NOT eight CUDA threads: grids have 2048×8 or6144×8 blocks; even one live expert supplies thousands of output rows. Sparse scheduling, scale scans and outstanding loads are plausible; occupancy/latency limitation is unproven.
`mmvq.cu:681` skips dummy weight reads when activation scales are finite; it still schedules blocks and checks scales. No nine-extra-record bandwidth should be charged.
## Trace distributions and ordering
µs [p10,p50,p90,p99]: IQ2 early single gate/up [125.056,160.192,179.904,197.856], late fused [34.336,71.712,145.055,195.680].
IQ3 down early [230.720,265.024,290.144,355.136], late [73.312,98.624,149.760,196.256]; IQ4 down early [171.584,238.080,323.040,328.000], late [28.736,94.944,160.992,210.528].
IQ3 gate/up exception early [119.840,174.208,233.344,234.528], late fused [35.712,134.880,236.928,292.096]; full min/max in audit.
Summing bank kernels through each down gives 4726 short layer occurrences: early mean577.371 µs [p10=462.752,p50=576.320,p90=639.072,p99=774.463]; late mean175.025 [107.648,170.432,296.000,415.296].
Long4800 occurrences: early mean531.203 [393.248,528.735,636.576,770.560]; late226.999 [107.808,230.432,366.880,498.943]; no explicit layer labels, so these are pooled layer occurrences.
For ALL >0.5ms kernel-union gaps: short1098 request→integer gather gaps=40.4580 ms/64;63 head→norm gaps=1.0976. Long2469=75.1974 plus63=1.1737.
Nearest bank before/after: short1024 early-down→late-gate/up gaps=24.2608 ms/64;74 pre-first-bank gaps=16.1972;63 token boundaries=1.0976. Long2469 early→late=75.1974.
Example ms from trace origin: early IQ2 ends3449.381; early down3449.382–3449.574; request3449.635–3449.637; gather3453.328; late fused3453.338–3453.525; late down3453.525–3453.696.
Thus early finishes BEFORE exposed wait; it cannot overlap an empty-GPU gap. Source split fill submission (`llama-moe-cache.cpp:4250–4282`) permits earlier I/O overlap, but CPU read start/end are not exported: hidden overlap is not measured.
## Copies and other kernels
Short cudaMemcpyAsync:1348 calls,21.2150 ms/64 host time,median2.720 µs,max64.125 ms; long0.2287 ms/64. Host API blocking time is not transfer-engine service time.
Trace:64 D→H copies of0.620 MB,0.0197 ms/64 device duration, matches154880 float logits;64 H→D0.066 MB,189 H→D0.025 MB;4914 D→D0.066 MB. Rounded copy byte fields limit precision.
Other small copies exist (audit); NO ~12 MB expert-record H→D copies. Maps are registered CPU-written memory (`llama-moe-cache.cpp:1924,4250`); normal bank reads directly consume host records. Neither slot-map uploads per layer nor DMA landing records explain21 ms.
Likely some host copy calls wait for preceding GPU work (e.g. logits readback); API sum lacks timestamps/directions/call stacks, GPU CorrId alone cannot join host durations. Exact21-ms attribution and critical-path fraction remain unresolved; do not budget21 ms savings.
“Other”15.3204 ms/64 top: float GEMV<1,256,false,false>4.8630; MMQ IQ2_XS2.5113; quantize_q8_1=1.7229; MMQ IQ3_XXS1.5558; argsort0.9423; MMQ Q5_K0.7589; bank_request0.5219; float add0.3878; MMQ Q8_0=0.3542; weighted reduction0.2376.
These sum13.8557 ms; rest1.4646. MMQ/prompt contamination explains much of short-minus-long “other”5.4100 ms; classifier also puts argsort here and rope_norm under norms.
## Window bg: rank 1 rejected as designed
`evidence/legs-bg2.txt`: all eight checks pass, but BG0=5.357/5.368 and BG1=2.137/2.136 tok/s (−60.2%); output md5 remains `9449ae5738`.
`evidence/BG1.log` explicitly says `GB10 bank gemv: on (plain vector loads, three groups; guarded host bank IQ)`.
The later BG gate failure is the missing **dense** GEMV policy log, not evidence that the bank schedule was off.
Use `GGML_CUDA_GB10_BANK_GEMV=0`; 0028 is a rejected experimental implementation, not a production optimization.

| Shape | positions/live | stride B | stock / bank µs | stock / bank useful GB/s | slowdown |
|---|---|---:|---:|---:|---:|
|IQ2 early gate/up|9/7|3637248|532.718 / 1549.574|47.79 / 16.43|2.91×|
|IQ3 early gate/up|9/7|4816896|392.397 / 2350.325|85.95 / 14.35|5.99×|
|IQ3 early down|9/7|4816896|427.294 / 1756.397|78.92 / 19.20|4.11×|
|IQ4 early down|9/7|6684672|285.469 / 3673.636|163.92 / 12.74|12.87×|
|IQ2 late fused gate/up|8/1|3637248|169.947 / 604.152|42.80 / 12.04|3.55×|
|IQ3 late fused gate/up|8/1|4816896|128.670 / 916.963|74.87 / 10.51|7.13×|
|IQ3 late down|8/1|4816896|103.093 / 295.678|46.72 / 16.29|2.87×|
|IQ4 late down|8/1|6684672|53.030 / 573.501|126.05 / 11.66|10.81×|

Stock bandwidth in this standalone bench is **not universally 130–160 GB/s**: useful bytes divided by stock_us gives 43–164 GB/s.
The historical engine aggregate ≈132 GB/s uses estimated live payload; it is a different measurement and denominator.
Copies rotate ≥512 MiB of live weights; late runs use 81–148 copies versus early 12–22. Gate/up late is fused (two weight tensors); down is unfused.

### What the code establishes, and what it cannot establish
`bank_bytes<N>` rounds the pointer DOWN to 16 B, uses `ld.global.v4.u32`, and fetches a second aligned vector when needed.
The quant fields may be only 2 B aligned, but the issued vector instructions are naturally aligned. Misaligned 16 B instructions are therefore **not** the explanation.
IQ2/IQ3 lanes consume 8 B slices spaced 8 B apart, in four quant blocks per warp; IQ4 consumes 16 B slices spaced 16 B apart, also four blocks per warp.
Those blocks are 74/98/136 B, so neither stock nor bank is literally one aligned 128 B line per warp instruction for every field.
Stock's small field loads nevertheless retain compact, adjacent lane accesses and immediate reuse of the same quant block's lines.
Rounding introduces overlapping lane requests and optional second instructions: over a complete block-offset cycle, vector-field requested bytes/useful field bytes are IQ2 2.75×, IQ3 3.333×, IQ4 1.5× (metadata excluded).
These are **lane request** ratios, not DRAM/C2C traffic ratios: coalescing and caches merge overlaps. The union of lines mostly covers the same contiguous block, with boundary rounding.
It is incorrect to equate that ratio with doubled host traffic, or to assert an order-of-magnitude increase in unique 64 B host lines.
CUDA sector counts, cache-line size, fabric transfers and host translation walks are distinct quantities; a translation walk is per translation miss, not per fetched line.

The three-group transformation also materializes partially populated quant blocks per lane, extracts bytes using runtime offsets, and extends their lifetimes before invoking unchanged dot helpers.
IQ4 has R=4 and 3×4×136=1632 B of source-level `bx` storage per thread; fused IQ2/IQ3 maintain both gate and up temporaries.
Scalar replacement may optimize this storage; dynamic byte selection can instead cause local-memory traffic, register pressure and reduced residency. No generated SASS, ptxas spill report or hardware counters are supplied to settle that.
Three source groups are not three guaranteed asynchronous memory groups: these are ordinary loads with dependent metadata/extraction, and the compiler controls their actual issue order.
More requests and longer reuse distances can increase system-path queue pressure, but overload of that path is a hypothesis, not a measured mechanism.
Thus the precise proven regression is this load/reconstruction/staging implementation; the evidence does **not** isolate a C2C hardware limit from instruction/spill costs.

The shapes discriminate against the proposed single causes:
- One-live late kernels remain at 10.51–16.29 GB/s, close to seven-live early 12.74–19.20: seven experts are not required for the collapse.
- IQ3 gate/up and down have identical stride and positions/live yet bank down reaches 19.20 vs 14.35 early, and 16.29 vs 10.51 late; K/N, fusion and work layout matter.
- IQ4 has the lowest vector-field inflation but the largest slowdown (10.8–12.9×), and stages four rows: rounding alone cannot account for the result.
- Larger IQ4 stride correlates with worse performance, but quant type and R change simultaneously. This is not an isolated stride or TLB experiment.
- Supplied THP observation (~160 GB/s with either page size) weighs against a page-size-only explanation; it does not measure translation misses in this kernel.

A viable bank schedule must preserve stock's compact warp-coalesced field/line access, near-term reuse, metadata reuse, dummy skip, accumulation and reductions.
Any prefetch must operate cooperatively on contiguous lines, avoid per-lane rounded overlapping gathers and local-block reconstruction, and justify its instruction/register cost.
A controlled one-group vs three-group comparison, stock-sized loads vs rounded loads, ptxas/SASS inspection and transaction/spill counters would be needed to assign the regression precisely; no further GPU experiment was possible here.

### Bandwidth ceiling and the read-only “A” comparison
The supplied registered-bank read benchmark's A result (~160 GB/s; earlier plan quotes 161.948) is a useful empirical streaming reference, not a proven architectural ceiling.
**Its source and full A output are absent from this sandbox**, so its exact lane width, row/stride mapping, reuse and cache policy cannot be verified locally.
`tools/uvmbw.cu` is a different benchmark: aligned `int4` array chunks, lane-consecutive elements, block partitions and a strided loop, with no IQ decode; its accumulator uses only x/y, so emitted load width needs checking too. It must not be relabeled A.
If A is a contiguous warp-coalesced bank scan, it shares stock's spatial locality principle, not stock IQ field instructions, scale lookups, dot arithmetic, dummy scans or reductions.
132/160≈82.5% suggests stock is already near the observed practical range for the aggregate path; reaching 160 would save at most 55.56−7333.4784/160≈9.73 ms/token under identical payload assumptions.
That is an optimistic bound, not an expected gain; neither “no schedule can beat stock” nor “132 is the hard C2C ceiling” follows. The 200 GB/s ceiling and prior 5–10 ms scheduling target are withdrawn.
GB10 has shared LPDDR; do not import discrete-HBM latency-hiding assumptions or claim a physically independent HBM tier on Spark.

## Revised ranking; gains overlap
1. **Former rank 1, three-group per-lane 16 B host IQ schedule: dead as designed.** Measured severe regression; no positive recoverable-ms budget. Preserve stock bank MMVQ. Cooperative line-preserving variants require new evidence, not another assumed gain.
2. **DMA ring for the ~1 missing record per layer: remaining experiment, high risk.** Keep resident early records on stock bank loads; after CPU fill, copy only the missing record to a reusable device allocation, then let late consumers wait on its event.
At ~0.917 GB/token, DMA alone costs ~5.73 ms at160 GB/s (4.58 ms at hypothetical200); it helps only if overlap and device-side read behavior repay the copy and synchronization costs.
The 12.92 ms late bucket is an optimistic whole-bucket upper bound, not a savings forecast; short-record overhead, shared LPDDR contention and fill completion timing may erase the gain.
Size slots for bounded in-flight misses, keep 64-bit maps and exact quant bytes, and prevent reuse until all consumers finish. The earlier two eight-record slots (~195.5 MB) are not a requirement for ~one miss/layer.
The old DMA bench mismatch must be fixed before interpreting throughput; no ring gain is validated.
3. **Sparse late launches/dummy scheduling: remaining experiment, medium risk; unvalidated target1–5 ms/token.** Avoid scheduling empty positions while preserving output indices, exact zeros and nonfinite-scale fallback.
One live expert still supplies thousands of row CTAs. Missing live-position metadata must be made graph-replay-safe; the map is filled asynchronously, so a CPU readback/synchronization can cost more than it saves.
Extra split launches alone at2–5 µs each suggest only0.30–1.13 ms/token; the whole late bucket is not removable launch overhead. Early fusion's launch-only budget remains0.15–0.375 ms, with register cost to measure.
4. **Allocation choice: lower-priority controlled experiment.** The earlier register/hostalloc numbers are read-bench observations, not an11.2 ms engine promise. Keep CPU fill visibility, memory pressure and address lifetime equivalent.
5. **Read readiness: separate I/O work.** Historical exposed waits remain relevant (short≈24.65 ms/token, long≈75.20); BG1's reduced waits are hidden by much slower compute and are not an improvement.
Acceptance: exact intermediate tensors and gold output, repeated off/on/off/on, same snapshot and prompt, measured live counts, decode-only trace, actual custom-launch evidence and joined copy/compute timing. No remaining gain is validated.

## Window gm and 0027 diagnostics
GM0=5.387/5.373; GM1=5.355/5.327 tok/s (−0.72% by two-run means; about−0.8%), gold. GM1 has **zero** `GB10 GEMV:` lines, including no `disabled (stock)` line.
0027 called its logger during `ggml_cuda_init` device enumeration, before the ordinary server log starts; surrounding CUDA enumeration logs are absent too. An early logger/filter issue is consistent with this, not proven by it.
The init guard silently returned on cc/SM mismatch, but did not control runtime dispatch. Absence of that line alone cannot establish that GM1 ran stock; treat this as an unvalidated/possibly stock run and the small delta as inconclusive noise.
The local Spark constant is1210; device12.1 maps to12*100+1*10=1210, and `sm_121a` does not introduce a different `__CUDA_ARCH__` integer. BG1 passes the same cc/48-SM/compiled-arch checks. There is no evidence identifying either guard as wrong.
Updated0027 moves policy logging to both runtime MMVQ entrypoints, before tensor eligibility; prints cc/nsm/compiled/requested and explicit rejection predicates; and logs the first actual custom launch per shape/device/thread.
Retain conservative guards and unchanged schedules. `win2gm.sh` is unchanged; its existing enabled-policy gate can now observe runtime logging. On rerun also inspect `GB10 GEMV dispatch:` to prove custom launches, because policy enabled alone is insufficient.
Rerun the full gm window on Spark after applying revised0027 plus the existing alignment patch; no valid dense performance verdict yet.
