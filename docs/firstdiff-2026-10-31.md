# First compute divergence, 2026-10-31
FD v2 completed four legs; its bounded streaming digest removed the memory failure.
FD10/FD20 are N=1/N=2; FD11/FD21 add MMVQ_EXACT_COLS=1.
The original window rejected a valid staggered prefill: column 1 has no step-0 output.
Samples are col0: 481,468,952 at steps 0,1,2; col1: 481,468 at steps 1,2.
Comparator aligns sample ordinals, requiring equal preceding tokens for decode inputs.
Thus column 1 step s compares against singleton step s-1; prefill intermediates are excluded.
Source: firstdiff2-check-r2jbtryu/src/llama-graph.cpp:2791-2918.
node_1 GET_ROWS gathers inp_mixed_tokens; node_2 DUP copies inp_mixed_embd;
SET_ROWS fills the alternate mixed embedding branch, then forward-select chooses input.
These are alternate-branch scaffolding, not position/cache rows or residual computation.
Only that exact four-node embedding prefix is classified as bookkeeping.
Both columns in both digest pairs first differ at indexer_k-0 after three matches:
embd GET_ROWS, norm-0 RMS_NORM, attn_norm-0 MUL.
The owner's count was six matches among 7056 aligned nodes for col0.
Source: src/models/glm-dsa.cpp:138,267: blk.0.indexer.attn_k.weight, [6144,128].
The output type f32 does NOT identify the weight type.
FD10-server.log:189 reports blk.78.indexer.attn_k.weight = 835584 bytes.
6144*128/32*34 = 835584: Q8_0; F16/BF16 would occupy 1572864 bytes.
Layer 0 is inferred Q8_0 from this same-role weight; its individual type is not logged.
Use tools/glm53/gguf_tensor_types.py on all GGUF shards to confirm without numpy.
For Q8_0, Spark dispatches MMVQ at BOTH N=1 and N=2, not MMVF/MMF.
mmvq.cu:1390ff chooses nwarps, small-K and singleton halve_iters by column count.
GB10's eligible singleton Q8_0 promotion doubles warps; N=2 uses its own specialization.
Different K partition/reduction association changes f32 bits within each column.
0008 splits ID MMVQ only (its condition is exact_cols && ids && ncols_dst>1).
0035 depends on 0008 and 0034 v2; it covers ordinary MUL_MAT, including non-ID MMVQ.
EXACT_COLS=1 recursively dispatches every width 2..8 as singleton, before FWHT/heuristics.
This covers float MMVF/MMF/cuBLAS and quantized MMVQ/MMQ/cuBLAS selection/geometry.
Packed temporary output and strided copies preserve higher-dimensional output layouts.
Exact mode disables CUDA fusion for both singleton and batch to match rounding boundaries.
Chain: ID MMVQ geometry -> ordinary Q8_0 MMVQ -> float MMVF/MMF/cuBLAS -> attention.
Expected next: DSA score/top-k (ties/visible KV extent), sparse/flash-attention tile/split
reductions, then possibly MoE split/merge/shared arithmetic; this is a prediction.
win2fd v3 reuses ~/llama-k3-par2/build-par2, runs FD30/FD31 with both exact flags.
Each server has a 300s lifetime cap; LEG-FAIL continues through WIN2FD-DONE.
No CUDA compiler/GPU here: application and host checks pass; Spark byte equality is pending.
