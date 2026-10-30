Patch 0008 is implemented, default off; no commits, pushes, network, or writes outside this sandbox.
Artifact: `0008-exact-mmvq-cols.patch` (raw workspace-relative patch; already applied here).
Apply to a fresh copy of this workspace with `git apply 0008-exact-mmvq-cols.patch` from its root, not from llama/.

Changes (paths relative to this directory):
- `llama/ggml/src/ggml-cuda/mmvq.cu:1361`: `GGML_CUDA_MMVQ_EXACT_COLS=1`, read once per type/process.
- `:1365`: only ID MMVQ with 2..4 columns; IQ2_XS, IQ3_XXS, IQ4_XS, Q2_K, Q3_K.
- `:1374`: dispatch existing singleton once per column: same device/type nwarps, small-K decision, K stride, partial-sum order, warp reduction and fusion.
- `:1372`: adjust fused shared-expert destination per column; keep expert-indexed weights/biases and byte bank strides.
- `:1384`: other MMVQ types retain original path with one warning per type. Non-MMVQ dispatch is not overridden.
- ncols=1 kernel/dispatch unchanged; quantize.cu:54/:557 unchanged: independent q8_1 blocks use identical activation arithmetic regardless of column count.
- Expected cost: 2..4 launches, repeated weight reads, no cross-column compute reuse; potentially several-fold compute/launch overhead. Cache union lookup unchanged.
- `tools/glm53/batchdump.cpp:40`: bounded K3XS v1/v2 header/index/record reader using public ggml API only.
- `:107`: synchronized CUDA MUL_MAT_ID graphs, singleton per column vs batches 2,3,4; optional SwiGLU graph.
- `:200`: same/overlapping/rotated routes, shared/per-route activations, raw F32 dumps and memcmp, unequal count/first mismatch hex.
- Off expectation requires a mismatch witness for EACH batch size; on requires every comparison identical. Failure exits nonzero.
- `tools/glm53/README.md:14`: Spark build/run instructions, input contract, limitations and detailed engine audit.

Spark commands (not executed here; rebuild patched CUDA libs at the requested library location first):
```sh
LIB="$HOME/llama-k3-bs/build/bin"
c++ -O2 -std=c++17 -Illama/ggml/include tools/glm53/batchdump.cpp -L"$LIB" -Wl,-rpath,"$LIB" -lggml-cuda -lggml -lggml-base -o tools/glm53/batchdump
EXPERTS=$(seq -s, 0 31)
GGML_CUDA_MMVQ_EXACT_COLS=0 tools/glm53/batchdump "$STORE" "$LAYER" up "$K" "$ROWS" "$EXPERTS" 8 1 seed off /tmp/up-off
GGML_CUDA_MMVQ_EXACT_COLS=1 tools/glm53/batchdump "$STORE" "$LAYER" up "$K" "$ROWS" "$EXPERTS" 8 1 seed on /tmp/up-on
```
Supply actual component K/ROWS (not encoded in store); repeat gate/up/down and all five types. Down uses CHANNELS=8, appropriate K/ROWS.
Repeat `swiglu` and `GGML_CUDA_DISABLE_FUSION=1` trials; supply X.f32 instead of seed for captured or identical-column activations.
Expected success lines, NOT measured results: `PASS: difference witnessed for each of batch 2..4 vs singleton columns`; `PASS: bit-identical batch 2..4 vs singleton columns`.

Engine verdict: NO exact greedy sequence guarantee for llama-server draft-mtp from this knob alone, including at temperature 0.
Remaining sites (following paths relative to llama/); numerical means potentially changed association/precision, not measured divergence:
- `ggml/src/ggml-cuda/fattn.cu:155/:180/:187`: query tile choice changes numerics. Need singleton tile/config, softmax/KV split/combine order and visible KV/padding per query.
- `ggml/src/ggml-cuda/fattn.cu:9/:128/:160`: sparse/dense and query-group union gather change traversal numerics. Need singleton sparse decision, ordered per-query key list and reduction.
- `src/models/glm-dsa.cpp:309`; `ggml/src/ggml-cuda/lightning-indexer.cu:618/:660/:700/:720`: fused 128x64/32/4 score paths keep the same per-query arithmetic at batch1..4; scheduling only (4-head tile starts at8, :240).
- `src/models/glm-dsa.cpp:319/:350`: unfused DSA KQ matmul can change numerics; upstream projections and top-k ties matter even fused. Need singleton projections/KQ, same visible KV/mask and tie behavior.
- `src/llama-graph.cpp:3166`; `ggml/src/ggml-cuda/mmvf.cu:811/:825`: v_mla column count can select MMVF vs MMF/cuBLAS, changing numerics. Need singleton dispatch/layout/precision per column; permute/cont only move bits.
- `src/models/glm-dsa.cpp:467/:489`; `src/llama-moe-cache.cpp:4970`; `src/llama-graph.cpp:2508/:2549`: predictor/prefetch/split guards disable batch operation. Local scheduling only: merge restores expert rows before original weighting/sum. Generalize union slot maps/events without changing arithmetic; remote computation excluded.
- `ggml/src/ggml-cuda/ggml-cuda.cu:2557/:2577/:2597/:2609`; `ggml/src/ggml-cuda/mmvq.cu:1409/:1447`: other dense projections/router/shared/head dispatch and non-ID MMVQ geometry can change numerics; need singleton dispatch/quantization/precision per column.
- `ggml/src/ggml-cuda/ggml-cuda.cu:2494/:4361/:4880`: fusion eligibility varies with shape; changed intermediate rounding can affect numerics. Need singleton fusion boundaries and rounding, not merely global fusion disable.

Local verification commands:
```sh
c++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only -Illama/ggml/include tools/glm53/batchdump.cpp
git -C llama diff --check
git apply --check --reverse 0008-exact-mmvq-cols.patch
```
Last output: `PASS: batchdump.cpp host syntax (-Wall -Wextra -Werror)`; `PASS: engine diff whitespace; patch reverse-apply check`.
Last output: `CUDA compiler: unavailable; CUDA compilation/runtime not tested`; `Patch: 375 lines`.
Uncertainty: no CUDA compiler, Spark, store/model or runtime proof available here. Template arguments/strides reviewed against surrounding code, not compiled as CUDA.
Harness does not exercise cache fill/eviction, large bank strides, shared-expert/bias fusion or dump internal q8_1. Off witnesses depend on nondegenerate data.
Source audit is not an exhaustive model trace; restored-KV intermediate/logit dumps and greedy token/rejection/rollback comparisons remain necessary. No speed or engine-exactness claim.
