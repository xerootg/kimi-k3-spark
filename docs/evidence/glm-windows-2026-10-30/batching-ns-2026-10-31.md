# Spark N-column decode — 2026-10-31
Local audit; window ns measurements supplied by owner, not rerun here.
Sources: `evidence/c-tree/ggml/src/ggml-cuda/{ggml-cuda,mmvq}.cu`,
`evidence/csv/glm-short_cuda_gpu_kern_sum.csv`, `streaming-decode-2026-10-31.md`.
This c-tree predates 0030/0032; its nested copy is duplicate evidence.
No mmq.cu, fattn*.cu, quantize.cu, norm.cu or rope.cu is available locally.

## Dispatch and exactness
| Operation at N=2/4 | Expected kernel / evidence | Exactness versus N=1 |
|---|---|---|
| Quantized dense projections, attention output, shared expert | `mul_mat_vec_q<type,2/4,...>` | Not guaranteed; changed K partition/reduction and fusion |
| Split host-bank experts | `mul_mat_vec_q<17/18/23,1,...>` per column (ns log) | Same arithmetic only for identical inputs, IDs, strides and split merge |
| Ordinary batch experts, split off | `mul_mat_vec_q_moe<type,...>` | Different reduction from one-column; not gold-guaranteed |
| Activation packing | `quantize_q8_1` family; `quantize_row_q8_1_cuda` call | Lossy; should be row-local and permutation-equivariant, implementation absent |
| Attention | `flash_attn_ext_f16` + `flash_attn_stream_k_fixup_uniform` likely | No bitwise guarantee across tiling/KV partition changes; exact N specialization unknown |
| Norm / RoPE | `rms_norm_f32`, fused norm/mul/RoPE variants, `rope_*` | Row-local for same input/position; fusion/geometry may change rounding; implementation absent |
| MMQ / cuBLAS | `mul_mat_q`, architecture GEMM names if fallback/prefill | Not expected solely because N=2/4; no bitwise equivalence guarantee |

`ggml-cuda.cu:2597–2653`: ordinary MUL_MAT tries MMVF/MMF for float types,
then MMVQ, MMQ, cuBLAS; bad padding or non-F32 activation/output can force cuBLAS.
`mmvq.cu:348–355`: GB10 Q2_K MMVQ limit is 6 columns, other quants 8.
Thus N=2/4 dense quantized work normally remains MMVQ, including attention output.
`mmvq.cu:1498–1537`: dense 2/4-column specializations; IDs + multiple columns
instead take the dedicated MoE launch (:1469); IQ3_XXS ID cap 7 admits both.
`ggml-cuda.cu:2537` restricts ordinary quantized dense fusion to one column.
`mmvq.cu:452–465,557–618,1421–1497`: N=1 can use small-K/wider blocks;
N=2/4 uses generic 4 warps and two rows/block. K summation can therefore differ.
`mmvq.cu:806–865` accumulates and reduces each j independently: no visible
cross-column sum or special arithmetic for column 0 versus column 1.
Q8_1 packing (:1941) takes row strides and dimensions; packing more rows alone
should not mix them. Quantization can amplify earlier drift but does not establish its origin.
Norm/RoPE should likewise permute with columns when positions and inputs match.
Attention must preserve sequence masks/KV offsets; total ctx 65536 also changes per-slot capacity.
N=1 CSV names above support attention expectations, not proof of the ns N=2/4 variant.

## Timing interpretation and next discrimination
Supplied steps ~175/345/380 ms imply a threshold-shaped cost, not its cause.
N=2 union 15.1/16 and 2.3 versus 1.3 missing records/layer imply substantial added
bank work/read demand. Early 6.7–7.7/8 completion does not measure late wait duration.
Dense fusion loss/launch geometry, attention tiling, graph updates, router/hooks,
and exposed read waits all remain candidates; kernel trace gaps cannot identify NVMe causally.
A 12 MB expert fitting in 25 MB L2 makes immediate same-expert reuse plausible,
but reuse distance, gate/up/down order and near-disjoint routing can evict it.
Flat N=2→4 step time does not prove free launches or L2 hits; counters/order are needed.
The design model predicted 267/444 ms (N=2/4), assuming fixed 68 ms spine and
bank/read scaling with union. Observed 345/380 invalidates that simple scaling;
changed predictions, contention and launch geometry prevent fitting a unique fixed cost.
Column-position MD5s implicate shared upstream/slot handling as well as split indexing.
Dense arithmetic can explain N-dependent drift, but its independent j loop alone does
not explain different results for identical columns at fixed N. Check first differing
intermediate tensor, masks/positions, packing strides, split maps/merge and scheduling.
N=4 instability can follow varying ubatch membership/order and divergent histories;
a race remains possible. Matching ordinary-batch MD5s weakens a split-only explanation.
M2 compares N=1, N=2 split on/off and N=4 split on, recording clients and GPU buckets.
M2 reuses win2m's prompt text; the separate 128-token workload text is not supplied.
Its capture includes prompt and draining edges; 31N decode-token passes are an estimate,
not measured ubatch steps. GPU sums and wall gaps must be read separately.
