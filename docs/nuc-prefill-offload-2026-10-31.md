# NUC prefill offload — design study, 2026-10-31
Scope: use both NUCs' RAM and compute for prefill; window-i4 hedge handles decode. No implementation proposed for immediate deployment.
Sources below are local; decimal MB/GB and dense-equivalent FLOPs (an FMA counts as two operations) throughout.
[A] Owner task: B=2048 tokens, K=8 routes, E=256 experts/layer, L=75 layers, D=6144, H=2048; resident n2=3588, n3=1471; Spark 62–67 tok/s, NVMe ~6 GB/s.
[B] `docs/experiments.md:1138,1160`: measured B=2048 prefill 65.9 tok/s; GH200 reads each nonresident expert once/ubatch, so bytes/token scale as 1/B.
[C] `tools/glm53/PLACEMENT.md`: NUCs own next-hottest records after Spark; exclusive RAM owners, mixed record sizes, decode-derived/synthetic placement is not a prefill distribution.
[D] `tools/glm53/EXACT-RESULT.md`, `FASTEXACT-RESULT.md`: scalar exact and parallel development; `MERGE-RESULT.md:34` supplies later validated exact-mode timings used below.
[E] `tools/glm53/cpu/SOL-RESULT.md:35` and `cpu/SOL-RESULT-2.md:24`: exact CPU ms/expert; latter is synthetic, laptop 165H, not deployed NUC hardware.
[F] `docs/experiments.md:629–635`: Spark path is shared 2.5 GbE; `evidence/FINDINGS.md:97–100`: Spark→NUCs measured 2.301/2.305 Gbit/s, faster NUC mesh does not bypass Spark port.
[G] `tools/glm53/g53x.cpp:377`: existing request = 17 int32 header + one D-float x, reply k D-float expert rows; `docs/remote-tier-2026-10-30.md:14`: RTT p50=1.57 ms.
[H] `evidence/0013-remote-tier-glm.patch`, `0013b-remote-hedge.patch`, `0013c-remote-hedge2.patch`: decode publication/fallback, unchanged Spark expert reduction; `evidence/remote-tier-result.md:11` describes row substitution.

## Work and existing compute baseline
Uniform routing model [A]: tokens/expert = B*K/E = 64; owned experts/layer m=n/L = 47.84 / 19.613 (nuc2/nuc3).
Per expert-token output (“row” here): F=3*2*D*H=75,497,472 FLOPs for gate/up/down; excludes quantization, SwiGLU, routing and transport [A,D].
Rows/layer R=64*m=3061.76 / 1255.253; FLOPs/layer R*F=231.155 / 94.768 GF; FLOPs/ubatch L*R*F=17.337 / 7.108 TF [A].
Rounded nuc2 check: m≈48 gives R=64*48=3072, R*F=231.928 GF/layer, times L=17.395 TF/ubatch [A].
Hotness means R=B*sum(p_e over owned e), generally ABOVE uniform prediction if decode hotness transfers; use measured prefill routes, not ownership fraction, for actual work [C].
Arc exact mode 2: L8 k=8 p50=13428 us [D, MERGE]; rate=8/0.013428=595.77 expert-token rows/s, F*rate=44.98 GFLOP/s dense-equivalent.
Thus GEMV replay compute/ubatch=L*R/rate=385.44 / 158.02 s; per layer R/rate=5.139 / 2.107 s [A,D]. Both run concurrently; nuc2 dominates.
That means B/max(compute seconds)=2048/385.44=5.31 tok/s before exposed LAN; NOT a forecast of a weight-reusing batched GEMM [A,D].
CPU exact real-kernel baseline: L8 k=8, eight workers, 3.748 ms/expert [E first]; rate=1/0.003748=266.81 rows/s=20.14 GF/s.
CPU compute/ubatch=L*R*0.003748=860.66 / 352.85 s; per layer=11.475 / 4.705 s; compute-only rate=2048/860.66=2.38 tok/s [A,E].
Newer synthetic CPU L75 k=8, twelve workers: 1.805 ms/expert [E second], 554.02 rows/s=41.83 GF/s; L*R*0.001805=414.49 / 169.93 s [A,E].
These are laptop proxies, not NUC prefill benchmarks. No batched exact GF/s is recorded. “~1.5x fast path” is not a measured GEMM rate: merge k=8 exact/fast=13428/15520=0.865 wall-time ratio on a loaded host [D].

## LAN: deduplicate input tokens, preserve expert outputs
Uniform distinct-route estimate: q(m)=1-product(i=0..K-1)((E-m-i)/(E-i)); using mean m approximates layer variation [A,C].
Probability token touches owner q=0.81375 / 0.47635; unique input tokens U=B*q=1666.56 / 975.57; input I=U*D*4=40.957 / 23.975 MB/layer [A].
All-token input ceiling=B*D*4=50.332 MB/NUC/layer. Hotness/correlation can push q toward one; use observed joint routes, not independent hit counts [A,C].
Exact reply must contain each expert-token row: O=R*D*4=75.246 / 30.849 MB/layer, NOT U rows; token-local NUC summing would change Spark reduction order [A,H].
An aggregated U-row reply would equal I bytes, but requires a separately proved reduction contract; no such byte-identity proof exists [H].
Per-NUC serial wire times (I+O)/(speed/8): at 1 Gbit/s =0.930 / 0.439 s/layer; at 2.5 Gbit/s =0.372 / 0.175 s/layer [A,F].
Both NUCs share Spark NIC: sum(I)=64.933 MB, sum(O)=106.095 MB; serial wire=sum(I+O)/(speed/8)=1.368 / 0.547 s/layer at 1 / 2.5 Gbit/s [A,F].
Full-duplex tiled streaming lower bound=max(sum(I),sum(O))/(speed/8)=0.849 / 0.340 s/layer at 1 / 2.5 Gbit/s [A,F]; cannot assume upload-whole-layer then reply achieves this.
At representative measured speed=(2.301+2.305)/2=2.303 Gbit/s [F], serial=171.028 MB/(2.303e9/8)=0.594 s/layer; ideal streaming=106.095 MB/(2.303e9/8)=0.369 s/layer.
IDs add ~R*two int32 mapping words plus U token indices, framing/status; not included in float-only bounds. Packet, copy and scheduling overhead also increase time [G, proposed protocol].

## Saving and critical path
Record fraction f=(3588+1471)/(75*256)=0.26349; assuming equal byte sizes, full ownership coverage and fully read-bound Spark, saved time=f*B/(62..67)=8.70..8.05 s/ubatch [A–C].
If remote entirely hidden: prefill=(62..67)/(1-f)=84.18..90.97 tok/s; central 65/(1-f)=88.25 tok/s. This is an upper bound, not a measured speedup [A,B].
Baseline layer T=B/65/L=0.42010 s; remaining read time S=(1-f)*T=0.30941 s [A]. Mixed sizes require byte-weighted f; other Spark compute/copies may impose another floor [B,C].
Ideal pipelined layer time=max(S,C2,C3,Wshared); rate=B/(L*time). Nonoverlapped chunks and layer barriers cost more; dependent layers cannot execute early [A,H].
Existing Arc GEMV C2=5.139 s dominates W: rate≤5.31 tok/s; existing CPU C2=11.475 s gives ≤2.38 tok/s (formulas above). Do not build network integration around these rates [D,E].
With sufficiently fast batched compute, LAN-only bounds: 1 Gbit/s serial/stream rates=2048/(75*{1.368,0.849})=19.96/32.17 tok/s; 2.5 Gbit/s similarly=49.89/80.43 tok/s [A,F].
At measured 2.303 Gbit/s, corresponding bounds=2048/(75*{0.594,0.369})=45.96 / 74.09 tok/s; streaming beats central baseline by 74.09/65-1=14.0% only under uniform routing [A,F].
To hide compute under S: nuc2/nuc3 need (231.155/94.768 GF)/S=747.08 / 306.29 GF/s, or R/S=9895.5 / 4056.9 rows/s; hotter routing raises these thresholds [A,D].
To stay under measured streamed wire W=0.36855 s instead: R/W=8307.7 / 3406.0 rows/s; compute/transport overlap and exact arithmetic must both be demonstrated [A,F].

## Protocol and exactness contract
Add negotiated PREFILL_EXECUTE framing: version, request/layer/ubatch ID, dimensions, arithmetic/store hash, lengths; unique token IDs + contiguous x rows; CSR expert→token/route-position lists.
Reply with request ID and (token ID, original route position, status, D-float output) per assignment. Bound allocations/lengths; preserve zeros/NaN payloads with explicit missing status, no sentinel ambiguity [G,H].
Alternatively PREPARE pins owned weights and reserves scratch; EXECUTE streams bounded token tiles. Cancel/drain stale requests; local fallback must retain access to skipped weights before layer reduction [H].
g53x needs expert-grouped tiled gate/up/down matrix operations, reusable quantized activation panels, bounded scratch, parallel CPU or iGPU scheduling and concurrent socket transfer; current GEMV launch batching alone does not reuse weights across tokens [D,G].
“Exact prefill” means identical g/u/h/down float bits and original route-position weighted reduction to Spark PREFILL MMQ for the same inputs, not equality to decode MMVQ or a tolerance [D,H].
Preserve MMQ quantization, scale rounding, FMA/partial accumulation order, FTZ/MUFU and NaNs; generic SYCL/BLAS GEMM reassociation is insufficient. Existing decode exact gates do not establish this [D,B].
Owner identifies 0021 as the layout-independent MMQ fix; no 0021 artifact is present here. Ledger [B:1164] explicitly records layout-dependent prefill. Verify its actual contract before adoption.
Before that fix, blanket md5 gating across changed cache layouts is invalid; a frozen-layout/dispatch stage oracle can still test a candidate, but decode md5 or matching text alone cannot certify prefill offload [B,D].

## Verdict and ordered milestones
Use NUC compute, but first build only an offline batched arithmetic benchmark after agreeing the MMQ oracle; do not ship unconditional all-owned offload. Existing GEMVs lose; potential streamed ~74 tok/s is conditional, not expected production performance [above].
Order: capture actual prefill per-layer routes/byte weights → adopt/verify 0021 and stage oracle → exact batched iGPU bench (CPU comparison) → local route scatter/reduction gate → framed tiled LAN bench → opt-in Spark skip-read integration and A/B.
First measurable milestone: owned expert-token row throughput with real records, actual token panels, all-stage bit comparisons and wall time including preparation; compare against 8308/3406 rows/s wire targets and 9896/4057 rows/s full-hiding targets (rounded formulas above).
Then measure both NUCs simultaneously over shared NIC, fallback frequency, saved NVMe bytes and end-to-end prefill; select profitable expert subsets if hotness makes reply bytes or compute exceed saved read time [A,C,F].
Serving weight BYTES is worse: 12 MB/(2.5e9/8)=38.4 ms (~40 ms) versus 12 MB/6e9=2 ms NVMe, ratio=19.2; even measured 2.303 Gbit/s gives 41.7 ms [A,F]. No record-byte proxy first step.
