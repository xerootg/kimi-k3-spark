Trace: 2048 tokens, 75 layers (3–77), 256 experts, top-8; evaluated tokens 1024–2047.
Inventory: {5: 2048, 4: 153600, 2: 153600, 3: 153600, 1: 151552}. Kind 1: 256 f32 predictor scores at layers 4–77; kind 2: 256 biased router f32;
kind 3: eight real i32 IDs; kind 4: 256 unbiased router f32; kind 5: one generated token i32. No kinds 6–9.
Kind 1 is the deployed predictor tensor; the binary does not encode its input-mode setting.
LRU: 80000 MiB; assumed 12,091,392 bytes/record (6937 slots); no index found.
Cold start; first 1024 tokens warm cache; shared issue budget 5 missing records/source layer.
a=previously resident, b=real consumed early read, c=real demand miss; a+b+c=600/token.
Useful=consumed speculative reads / speculative reads; total reads=demand+speculative.
Instant completion; no prefill cache, pinning, async cancellation or partial-read timing is modeled.

|Policy|reads/tok|PF/tok|useful|a/layer|b/layer|c/layer|evict/tok|PF evict/tok|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|none|147.28|0.00|0.0%|6.036|0.000|1.964|147.28|0.00|
|top8|196.38|154.86|70.5%|5.990|1.456|0.554|196.38|154.86|
|top10-cut|233.81|202.48|59.2%|5.984|1.598|0.418|233.81|202.48|
|sticky-union|196.38|154.86|70.5%|5.990|1.456|0.554|196.38|154.86|
|sticky-rerank|179.81|120.28|75.2%|6.000|1.207|0.794|179.81|120.28|
|prior-rerank|193.26|150.51|71.4%|5.997|1.432|0.570|193.26|150.51|
|prior-only|150.35|3.80|2.9%|6.045|0.001|1.954|150.35|3.80|
|two-ahead-spare|196.38|154.86|70.5%|5.990|1.456|0.554|196.38|154.86|
|top8-cut|168.97|104.90|80.9%|6.014|1.132|0.854|168.97|104.90|

Top-8 precision = recall: 79.08%; previous-token same-layer reselection: 21.74%.
Baseline c=0.554/layer versus Spark ~1.35; difference -0.796, not fitted.
Per-layer precision/recall/reselection and a/b/c: evidence/pfstudy/layers.csv; per-token counts: tokens.csv.
top10-cut adds ranks 9–10 only within 10% of |score8|; sticky-union appends previous-token IDs.
sticky-rerank boosts sticky scores by .15*(score1-score8); prior-rerank by .5*span*rolling frequency.
Prior uses last 64 tokens only. top8-cut keeps six candidates. Constants fixed without evaluation tuning.
two-ahead-spare uses PREVIOUS-token IDs for L+2 after next-layer jobs within the SAME five-read budget.
Current-token L+2 kind-1 scores do not yet exist at L: using them would leak future computation.
Sticky union and L+2 proxy add no reads: previous-token IDs remain cached. Extra L+2 occupancy: 0 record-layers/token.
Ranking by c records saved per extra total read (negative extra reads means a saving):
- top10-cut: saved +10.184/token; extra reads +37.433; ratio 0.272
- two-ahead-spare: saved +0.000/token; extra reads +0.000; no exposure improvement
- top8-cut: saved -22.551/token; extra reads -27.411; no exposure improvement
- sticky-union: saved +0.000/token; extra reads +0.000; no exposure improvement
- sticky-rerank: saved -18.014/token; extra reads -16.570; no exposure improvement
- prior-rerank: saved -1.238/token; extra reads -3.115; no exposure improvement
- prior-only: saved -105.030/token; extra reads -46.031; no exposure improvement
Recommendation: none beats baseline exposure without more reads; retain input 3 and instrument before patching.
Patch sketch: in llama-moe-cache.cpp, rerank before prefetch_decode reservations/spec_reclaim; retain the five-miss cap.
Use fixed per-layer arrays (256 score/count entries, eight previous IDs, rolling-ID ring); allocate once at model load.
Update history after real fill/prediction_measure; skip resident IDs without promoting them; keep existing single-token guards.
For chosen reranker apply the exact formula above; otherwise trial sticky-rerank behind an env flag, with PF-used/cancel counters.
Trace precision differs from window p (79.08% vs 85.2%); cannot identify input mode, input-4/5 alternatives, genuine L+2 scores, completion deadlines, or initial cache/prefill state.
Window-p legs: input3 85.2% precision/67.7% PF useful; input4 85.5%/65.5%; input5 41.9% PF useful.
Input5 62.7% aggregate is candidate precision (12 IDs), not top-8; shown layers 75–77 recall 90.72–98.43%.
place.log used synthetic Bernoulli predictions when missing; its fitted baseline is not independent validation.
