# Host-bank GEMV: evidence correction and experiment plan
Local evidence only; no kernel patch, GPU run, network, or claim of measured optimization gain.
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
## Ranked design, estimated recoverable ms/token; gains overlap
1. **Host IQ load scheduling (medium risk): screening target5–10 ms**, ideal whole-bank ceiling55.56−7333.4784/160=9.73 ms, /200 gives18.89 ms (63-pass baseline increases ceilings≈0.9ms).
Port the scheduling concept, not Q5_K arithmetic: IQ2/IQ3/IQ4 decode and vector-dot layout differ. Keep each lane's block assignment, accumulation sequence, warp/block reduction, fusion and merge unchanged.
Compare plain aligned16B loads with2–3 groups in flight, pipe3-16B, then bulk-R4-B8. Gather each expert base once, tile output rows within its record, preserve IDs/zero-slot/shared exceptions and 64-bit strides.
Host path is explicitly excluded by0027 (`!ids && !fusion && ...dense_weight`); its dense correctness/performance results do not validate host IQ schedules. Uncached records need enough independent requests; larger tiles can hurt sparse occupancy/register pressure.
`cp.async.cg.shared.global` uses a global source address and bypasses L1, but these local artifacts do not demonstrate mapped-system-memory support/efficiency on GB10. Test actual registered/hostalloc addresses, aligned16B tails, visibility/fences and sanitizers; plain vector-load path is the control/fallback, not an assumed loser.
Bulk TMA and per-thread cp.async are distinct paths; test both separately. GB10 has shared LPDDR, not discrete HBM; device staging does not create an independent high-bandwidth weight store.
2. **Sparse/dummy and launch scheduling (medium risk): target1–5 ms**, not a measured gain. Late minima≈34/73/28 µs (IQ2 fused/IQ3 down/IQ4 down) imply ~7.9 ms/token floor if representative of empty launches; IDs are absent, so not proven all-dummy.
Check live-count histograms; skip fully empty late work or schedule only active positions while emitting exact zeros/scratch output. Keep dependencies and graph replay viable; CPU count readbacks could erase gains.
Extra split launches≈150–225/token versus unsplit, at assumed2–5 µs overhead gives0.30–1.13 ms; most12.92 ms late time is not launch overhead alone. Batch consecutive output rows within a layer; cross-layer batching violates activation dependencies.
Test fused EARLY gate/up (currently two launches) for up to75 fewer launches≈0.15–0.375 ms launch-only plus load-scheduling benefits; extra live registers can reverse the gain. Never change arithmetic order.
3. **DMA ring for late records (high risk): plausible0–5 ms**, optimistic ceiling≈12.9 ms late bucket; at~1 miss/layer,0.917 GB/token DMA costs4.58 ms at200 GB/s; each12.22 MB record≈61.1 µs.
Pipeline DMA after CPU fill completion in another stream while early resident work runs (~577 µs/layer); late GEMV waits on ready event and uses device addresses. If fill ends only at exposed wait, its DMA is additional visible work, not hidden.
At hypothetical device GEMV200 GB/s, late read service another4.58 ms: serialized total9.17 vs12.92 saves3.75 ms; fully hidden DMA leaves4.58 ms and saves8.34 ms before fixed overhead. Unified-memory contention and sparse floor reduce both estimates.
Two8-record ring slots≈195.5 MB plus maps/events; never recycle until all consumers finish. Copy exact quantized bytes and preserve arithmetic/ID order. Repair bench DMA variant mismatch first; its~200 GB/s is an assumed target, not validated ring performance.
4. **Allocation choice (medium operational risk): ceiling55.56×(1−161.948/203.011)=11.24 ms**, only if actual workload scales like repeated-bank bench; compare register versus cached hostalloc on same snapshot, CPU fill/reads and memory pressure. Not additive to rank1 bandwidth ceiling.
Payoff ranking within GEMV: scheduling ceiling9.7–18.9 ms; allocation sensitivity11.2; sparse scheduling target1–5; DMA target0–5 (optimistic8.3). Risk ranking: plain-load scheduling < allocation < compaction/fusion < DMA ring; all require measurement.
5. **Read readiness (I/O work, outside GEMV):** short exposed decode hook gaps≈24.26×64/63=24.65 ms/token versus long75.20;10–25% improvement means2.5–6.2 ms short,7.5–18.8 long, conditional targets. Read start/completion plus NVTX layer/token markers needed.
Acceptance for every candidate: exact intermediate tensors and output hash on real/dummy/NaN-scale cases; frozen snapshot/prompt, repeat short+long, pure-decode ranges, live-row counts, bytes and joined CUDA API trace. Reject speed at changed accumulation order. No speedup validated here.
