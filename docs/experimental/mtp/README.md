# Exact expert batch experiment (patch 0008)

`GGML_CUDA_MMVQ_EXACT_COLS=1` is read once per MMVQ type, default off; use a new process for each mode.
For MUL_MAT_ID with 2..4 columns and IQ2_XS, IQ3_XXS, IQ4_XS, Q2_K or Q3_K, dispatch each column through the existing ncols=1 dispatcher.
This preserves the device/type-dependent singleton nwarps (four on GB10 for these normal expert shapes), small-K decision, K stride, ordered shared-memory additions, warp reduction and fusion.
Activation q8_1 quantization is unchanged: quantize.cu:54 reduces independent 32-element blocks; batch count changes grid.z and addresses, not arithmetic within a block.
Quantized activation column offsets are in block_q8_1 units; IDs are in int32 units; outputs are in float units; bank offsets remain byte strides.
Shared expert weights stay fixed and shared_dst advances by shared_stride_col_dst per column. Biases remain expert-indexed, not token-indexed.
The ncols=1 dispatch and kernel are unchanged. Unsupported MMVQ types retain original dispatch and emit one warning per type when a gated-size call is encountered.
Batches outside 2..4, non-ID matmuls, and operations that never select MMVQ are unchanged; this knob does not force MMVQ on other devices.
Expected cost: 2..4 singleton kernel launches instead of one MoE launch, repeated weight reads, no inter-column compute reuse, potentially several-fold compute/launch overhead. Store/cache union lookup is unchanged.

## Build and run on Spark

Rebuild the patched CUDA libraries in ~/llama-k3-bs/build first. Do not link an unpatched installation.
From this sandbox directory (runtime loader must also resolve the CUDA dependencies of libggml-cuda):

```sh
LIB="$HOME/llama-k3-bs/build/bin"
c++ -O2 -std=c++17 -Illama/ggml/include tools/glm53/batchdump.cpp \
    -L"$LIB" -Wl,-rpath,"$LIB" -lggml-cuda -lggml -lggml-base -o tools/glm53/batchdump
# Set STORE, LAYER, K, ROWS for an actual component. Store metadata has byte sizes/types, not dimensions.
# Example route bank; load at least 32 records for four disjoint groups of 8.
EXPERTS=$(seq -s, 0 31)
GGML_CUDA_MMVQ_EXACT_COLS=0 tools/glm53/batchdump "$STORE" "$LAYER" up "$K" "$ROWS" "$EXPERTS" 8 1 seed off /tmp/up-off
GGML_CUDA_MMVQ_EXACT_COLS=1 tools/glm53/batchdump "$STORE" "$LAYER" up "$K" "$ROWS" "$EXPERTS" 8 1 seed on /tmp/up-on
```

Arguments: STORE LAYER gate|up|down|swiglu K ROWS EXPERT_CSV USED CHANNELS seed|X.f32 off|on DUMP_PREFIX.
K is input width, ROWS output width. Each store record contains packed gate|up|down, with padding only after down.
The selected component's computed byte size must match the index. EXPERT_CSV maps loaded records to compact GPU slots; IDs select those slots.
CHANNELS must be 1 (shared activation, gate/up) or USED (per-route activation, down). No repeated expert IDs within a column.
`seed` creates deterministic nonzero F32 activations; X.f32 supplies exactly 4*K*CHANNELS native F32 values in [column][channel][K] order.
Repeat for gate, up and down on all five types/layer layouts; down uses its transposed K/ROWS and CHANNELS=USED.
`swiglu` loads up and gate, constructs two MUL_MAT_IDs followed by ggml_swiglu_split; normal backend fusion may combine them.
Repeat with `GGML_CUDA_DISABLE_FUSION=1` in both processes to exercise unfused operations. The tool does not assert that fusion was selected.
For identical-input trials, supply X.f32 with one column repeated four times; the shift=0 case then repeats IDs too.

For each route shift (0, 1, USED), the harness runs four independent singleton graphs and batches of size 2,3,4 with identical input/ID bytes.
It synchronizes CUDA, compares every output F32 bit using memcmp, prints unequal count plus first mismatch index/hex words, and writes packed binary dumps.
Exit 0 in `off` mode requires a mismatch witness for EACH batch size across the route patterns; exit 0 in `on` mode requires ALL comparisons identical.
Exit 1 means the expectation failed; exit 2 means usage, store, allocation or compute failure. Equal zero/degenerate inputs may fail the off witness test without indicating a harness bug.
The final expected success lines are `PASS: difference witnessed for each of batch 2..4 vs singleton columns` and `PASS: bit-identical batch 2..4 vs singleton columns`.
For external cmp, compare the first N*ROWS*USED*4 bytes of shiftS.single.f32 against shiftS.batchN.f32 (the singleton file contains four columns).
Passing both modes on the same corpus demonstrates this operator issue and its correction on that build/device; it is not an engine or universal proof.
This harness uploads compact packed banks; it does not exercise large bank strides, cache fills/evictions, fused shared experts, bias fusion, or dump internal q8_1 bytes.
Those require engine dumps/additional cases on Spark. File byte counts/offsets are checked before reads; malformed model payloads are not validated as quant blocks.

## Engine exactness audit (paths relative to llama/)

No: enabling this knob does NOT establish that draft-mtp verification emits the sequential batch-1 greedy token sequence.
Temperature 0 only makes sampling greedy; any changed logit or router/indexer tie can change the selected token and all subsequent state.
The following remaining batch-sensitive paths need separate gates/dumps; no implementation is included here:

| Site | Per-column numerical effect and required singleton-equivalent variant |
| --- | --- |
| ggml/src/ggml-cuda/fattn.cu:155, :180, :187 | Query count selects MMA query tile 1/2/4 etc. (depending on GQA). Tile config affects reduction/softmax association, not just scheduling. Pin singleton tile/config/precision for every query, including KV split/combine order and visible KV extent/padding. |
| ggml/src/ggml-cuda/fattn.cu:9, :128, :160 | Sparse versus dense and grouped sparse-union gather change traversal and tile padding. Use singleton per-query sparse decision and ordered key list, same mask and reduction. Causal masking alone does not prove bit equality. |
| ggml/src/ggml-cuda/fattn.cu:634 | DKQ>256 excludes vector FA; 576/512 MLA stays MMA, but that does not remove the preceding tile/sparse differences. Exact model dimensions remain to be confirmed from GGUF. |
| src/models/glm-dsa.cpp:309; ggml/src/ggml-cuda/lightning-indexer.cu:618, :660, :700, :720 | Fused DSA: 128x64/32 keep WMMA (or vector for F32/BF16), 128x4 stays vector until threshold8 (:240). Batch1..4 changes grid scheduling only for these score kernels; no new exact kernel needed there. Compare upstream Q/K/weights, visible scores and top-k ties/KV padding. |
| src/models/glm-dsa.cpp:319, :350; src/llama-graph.cpp:3588 | Unfused DSA KQ matmul may select different arithmetic as query count grows; ReLU/weight/sum/mask/top-k then propagate it. Replay singleton matmul with singleton precision/strides, same visible KV and deterministic top-k tie behavior; retain per-position causal mask. Fused/unfused selection itself is a context flag, not a batch-count branch. |
| src/llama-graph.cpp:3166; ggml/src/ggml-cuda/ggml-cuda.cu:2577, :2597, :2609; ggml/src/ggml-cuda/mmvf.cu:825 | v_mla puts tokens in matrix columns. F16 singleton MMVF can become MMF/cuBLAS for 2..4; F32 MMVF threshold is 3 (:811). Accumulation/precision can change. Replay each column through the singleton dispatcher with identical layout/precision. Permute/cont alone only move bits. |
| src/models/glm-dsa.cpp:467, :489; src/llama-graph.cpp:2508, :2549; src/llama-moe-cache.cpp:4970 | Predictor/prefetch and its split path require batch1. Local prefetch only changes cache scheduling. Split restores each expert row by position before the original weighted sum, rather than summing hit/miss groups. Generalization needs union routes/slot maps, correct events/lifetimes and unchanged row arithmetic; remote expert execution is outside this local exactness claim. |
| ggml/src/ggml-cuda/ggml-cuda.cu:2557; ggml/src/ggml-cuda/mmvq.cu:1409, :1447 | All ordinary dense projections (attention, router, shared experts, output head) retain MMVF/MMF/MMVQ/MMQ/cuBLAS batch heuristics; even non-ID MMVQ small-K/halve_iters vary with columns. Replay singleton dispatch and geometry per column, matching precision and activation quantization. This knob intentionally covers only ID MMVQ. |
| ggml/src/ggml-cuda/ggml-cuda.cu:2494, :4361, :4880 | Fusion eligibility can change with batch shape, particularly dense GLU (singleton-only) and routed/shared matching. Fused vs unfused intermediates may change rounding. Match singleton fusion boundaries and intermediate rounding per column, rather than merely turning all fusion off. Routed ID fusion is retained by this patch when selected. |

This is a source audit, not an exhaustive measured trace of the actual model graph. After closing these paths, compare logits and intermediate tensors from restored identical KV snapshots, then compare baseline/MTP greedy IDs with rejection/rollback cases.
