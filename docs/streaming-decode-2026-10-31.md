# Streaming decode on GB10 — 2026-10-31
Scope: one short-context GLM-5.3 decode; local evidence only, no new GPU run.
Sources: `evidence/csv/glm-short_cuda_gpu_trace.csv`, `evidence/breakdown-v2.txt`,
`evidence/host-bank-gemv-plan.md` (especially Outcome), and `evidence/c-tree/`.
“O(1)” means bounded steady service per token at fixed context/cache state; attention and misses still change with context.
## One layer today
CSV has kernels, grids and timestamps, but no layer names, tensor addresses, routing IDs or NVMe events.
This is a representative ordinary IQ2/IQ3 layer occurrence, not a named layer or an average; times are µs.
Absolute starts below are ms from capture origin. GPU weight traffic uses shared LPDDR, including the C2C host path.
| Phase / start ms | Duration µs | Busy resource | Idle / overlap |
|---|---:|---|---|
| Router 9100.609152; top-k 9100.638528 | 29.568 + 3.904 | GPU / LPDDR | Bank GEMV idle; NVMe may still speculate |
| Early request 9100.642496 → map 9100.676480 | 33.984 elapsed; request 1.984 | CPU poller plans slots, starts missing reads | GPU waits for map; NVMe fills if needed |
| Shared gate/up 9100.677760; down 9100.766912 | 88.064 + 49.600 | GPU / LPDDR | C2C expert GEMV idle; fills can run concurrently |
| Resident gate 9100.818560; up 9100.985408 | 166.144 + 179.936 | GPU / C2C / LPDDR | NVMe can fill misses; no next-layer attention |
| Resident down 9101.166784 | 284.160 | GPU / C2C / LPDDR | Same overlap; expert activation needs gate/up |
| Reduction, predictor, real-hook region 9101.450944 → 9101.571200 | 120.256 elapsed | GPU bookkeeping/predictor, CPU hook | Contains GPU work: NOT 120 µs of pure I/O wait |
| Late fused gate/up 9101.571200 | 35.520 | GPU / C2C / LPDDR | Miss data now ready; NVMe may speculate ahead |
| Late down 9101.607360 | 73.343 | GPU / C2C / LPDDR | Next-layer attention still blocked |
| Merge/residual → next-layer projections 9101.686912 | merge ≈6; first projection 50.080 | GPU / LPDDR | Bank consumers finished; NVMe may run ahead |
| Next attention: projections → flash 9101.991840; fixup 9102.010432 | flash 18.240 + fixup 2.080 | GPU / LPDDR | C2C expert GEMV idle; NVMe only if queued |
Shared executes BEFORE resident GEMVs here: its independent branch hides fills, not simultaneous GPU weight streaming.
Thus the requested phase list is not a strictly serial shared-after-down schedule.
Pooled layer bank work: early 577.371 µs (gate/up/down), late 175.025 µs; 4726 occurrences.
IQ2 early gate/up mean 156.189 µs each; IQ3 early down 264.953; late fused 73.746, down 101.359.
Gaps >0.5 ms: early→late 24.261 ms/captured token (24.646 steady), including 20.959 after 6144-wide down.
Pre-first-bank gaps add 17.295 ms/captured token, mostly 16.197 with no preceding bank: capture/prompt pollution.
These are temporal wait candidates, not measured NVMe latency. Router/shared meanings above follow shape/order; no semantic tags.
Capture: bank 55.560, spine 68.458, GPU interval union 141.398, all gaps 51.956 ms/token: 193.355 ms (5.17 tok/s).
There are 63 decode passes for 64 normalization tokens; steady bank/spine estimates are 56.442/69.545 ms, prompt unremoved.
## What one sequence can overlap
Dependency: output L−1 → attention L (including projections) → router L → actual expert IDs → missing reads
→ late gate/up → late down → merge with resident/shared contributions → attention L+1.
Resident gate/up/down can hide reads after the early hook; late down cannot precede its late gate/up.
Predictor 0005 applies the next router to an earlier activation: speculative reads already bypass the actual-router dependency.
Shared expert depends on the layer input, not routed IDs; it already covers part of the fill interval.
Attention L+1 cannot start early. Its weights are known, but prefetching ~180 MB/layer into 25 MB L2 cannot retain the spine.
Small tiles might prefetch near use; this does not remove the full next-layer weight stream, and competes for shared LPDDR.
A trace-conditioned chain budget is ~68+56+24.6+10 = 158.6 ms (~6.3 tok/s): retain exposed real-hook waits,
remove off-chain overhead only. Versus supplied ~174 ms warm, headroom is ~15 ms (~10% throughput), not a 2× win.
This is an estimate, not an identified physical minimum: the ~10 ms remainder and irreducible portion of waits are not tagged.
Using ALL observed GPU work conservatively instead gives 141.398+24.261=165.659 ms (6.04 tok/s),
27.696 ms below the profiled 193.355 ms; do not compare that profiled delta directly with unprofiled warm latency.
If perfect prediction removed every exposed read wait, the rounded compute chain alone is ~134 ms (7.46 tok/s).
That is a stronger hypothetical, not overlap of independent work already available in this trace. Keep the outcome's ~5.8 warm reference.

## N sequences per ubatch: explicit capacity model, not a benchmark
`--parallel N` needs N active clients and an ubatch containing their tokens; idle slots do not batch work.
Dense spine becomes N-column matrix work: same weight bytes, N× FLOPs; N=2–4 should be cheap while bandwidth-bound.
No routing IDs are exported. Assume independent uniform 8-of-256 selections: D(N)=256[1−(31/32)^N], r=D/8.
Assume constant miss fraction: distinct misses M=1.35r/layer; bank B=56r ms, exposed waits W=40M/1.35=40r ms.
Plain serialized budget T=68+B+W+10=78+96r ms. Expert-union reuse is an optimistic implementation assumption.
To enforce 13 GB/s, ALSO budget total reads: supplied 9 GB/s × rounded 0.174 s = 1.566 GB/N=1 step,
including the 61% speculative traffic. Assume these bytes scale with r, unchanged prediction usefulness/cache capacity.
NVMe service I=1000×1.566r/13=120.462r ms; T_effective=max(T,I), avoiding double-counting exposed W.
| N | distinct D | misses M/layer | spine / bank / wait ms | T ms | I ms | aggregate tok/s |
|---|---:|---:|---|---:|---:|---:|
| 1 | 8.000 | 1.350 | 68 / 56.00 / 40.00 | 174.00 | 120.46 | 5.75 |
| 2 | 15.750 | 2.658 | 68 / 110.25 / 78.75 | 267.00 | 237.16 | 7.49 |
| 4 | 30.531 | 5.152 | 68 / 213.72 / 152.66 | 444.37 | 459.73 | 8.70 |
| 8 | 57.421 | 9.690 | 68 / 401.95 / 287.10 | 767.05 | 864.63 | 9.25 |
NVMe is the sole limiting service bucket from N=4 IN THIS MODEL; 40r alone never establishes that crossover.
Without the total-read bound the estimates are 5.75/7.49/9.00/10.43 tok/s. No measured saturation N exists here.
Skew/correlated clients lower D; cache churn or worse prediction raises reads. A fixed 13 GB/s is supplied, not CSV-derived.
LPDDR contention and actual per-token bank rereads can invalidate the independent-service model.

## Changes beyond --parallel
Source paths below are relative to `evidence/c-tree/` (the nested duplicate is not a separate implementation).
`src/llama-moe-cache.cpp:5163` get_split_map rejects n_tokens!=1: N>1 uses ordinary batch fill, without early/late maps.
`ggml/src/ggml-cuda/ggml-cuda.cu:2685` dispatches small quantized MUL_MAT_ID to ggml_cuda_mul_mat_vec_q;
`ggml/src/ggml-cuda/mmvq.cu:1469` routes IDs with ncols_dst>1 to mul_mat_vec_q_moe_launch, not the one-column kernel.
Important correction to round 12: that kernel ALREADY has finite-scale guarded zero_slot skipping (:1196; metadata :1914).
But ordinary batch maps do not establish marked split views: the current multi-token route loses split overlap.
IQ3_XXS MMVQ caps at 7 columns (:177); N=8 exceptions take subsequent MMQ/MMF/fallback dispatch, not uniformly this kernel.
Extend split graph/maps/merge to N columns, deduplicate expert IDs, preserve per-column outputs, mark zero slots,
and group columns by expert to reuse weight loads; existing per-token warps do not guarantee one physical read per union expert.
Predictor (:4689) expects 8 (or wide 12) IDs, discarding other counts; batch fills discard predictions (:4596).
Need one prediction per sequence/layer, union reads, measured union recall, bounded reservations and graph-safe publication.
Two ubatches on two streams could run A's next layer while B waits, but today's single graph per ubatch does not interleave them.
Serialize bank hooks; replace global split_il/maps/predictor state with in-flight state, pin slots until both consumers finish.
Ideal hiding saves at most W: N=1 174→134 ms (1.30×), N=2 267→188.25 ms before the I/O floor.
With I included, N=2 bottoms at 237.16 ms (8.43 aggregate, only 1.13× vs plain batching); N≥4 has no modelled gain.
Shared LPDDR also prevents assuming independent spine/C2C maxima. Defer this complex pipeline until batching evidence warrants it.

## Window par decides
Already queued: --parallel 1/2/4, concurrent clients, aggregate decode tok/s and per-client MD5 against gold.
Record actual ubatch token counts, per-client latency/fairness, routing union/misses, prediction usefulness, read bytes and NVMe busy/GB/s.
Trace bank dispatch, split/dummy eligibility, hook waits and LPDDR traffic: correctness plus throughput alone cannot prove overlap.
Decide whether spine amortizes, ordinary batch fills regress, or NVMe saturates near N=4; then prioritize N-column split/union reuse.
Only consider two in-flight ubatches if substantial exposed waits remain while NVMe and LPDDR have spare service capacity.
