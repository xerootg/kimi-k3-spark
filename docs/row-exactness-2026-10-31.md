# GLM decode row exactness — 2026-10-31

Scope: N=1..8, equal input bits, width, positions, KV visibility, strides/alignment and fusion boundaries.
Sources: supplied evidence/c-tree ggml-cuda.cu/common.cuh and c-tree-extra CUDA leaves; references below are CUDA-relative.
Missing production leaves are examined in llama/ggml/src/ggml-cuda and explicitly marked LOCAL; they are not proof of the installed GB10 implementation.
The reported fd6 observer equality removes production fusion boundaries; it does not prove that a particular fused kernel changes arithmetic with N.
fdg3 outcome is not available here. No CUDA compiler/GPU or production launch log is available.

| Kernel / source | Dispatch depends on N? | Identical per-row arithmetic? |
| --- | --- | --- |
| RMS_NORM, norm.cu:399–486 | Grid yes; block/vpt/vector specialization no direct row-count predicate. Width is ne00, NOT decode ncols=N. | Yes with same alignment/operand packing. Vector n4=width/4 chooses (32,128,256,512,1024), vpt=4 only for 1024<n4<=4096; scalar uses 256 below width 1024, else 1024. |
| Fused RMS+MUL(+ADD), RMS+SCALE, norm.cu:489–591 | Same dispatcher as RMS; MUL/ADD width and alignment also gate vectorization. | Same N=1/N>1 specialization has the same square sum/reduction tree and ordered norm→mul→add or norm→scale epilogue. Retained. Scalar mul+add may contract; this is not N-dependent within a specialization. |
| LayerNorm/indexer NORM, norm.cu:358 | Grid only; threads=32 for width<1024, otherwise 1024. | Yes: fixed float2 sum(x),sum(x*x) tree; var=E[x²]−mean². No reduction across tokens. |
| ROPE normal/neox, rope.cu:358/404 | Fixed (1,CUDA_ROPE_BLOCK_SIZE); N changes grid/indexing. | Yes, independent pairs and supplied positions/frequencies; no cross-row reduction. |
| Fused RMS+MUL+ROPE(+SET_ROWS), rope.cu:709/801 | Threads=256 below width1024, else1024; grid only depends on rows. | Yes for the same fusion: scalar square-sum tree, then norm→weight→pair rotation→optional scatter. This tree can differ from vector RMS when fusion boundaries differ. |
| ADD/MUL/DIV and fused multi-ADD/MUL, binbcast.cu:31/177 | Yes: collapse, block xyz and unravel fallback depend on shape/contiguity. | Same element operation; both kernels fold src operands left-to-right using a comma fold. No N-selected reassociation. Supplied leaf has no float4 arithmetic branch; vector adds elsewhere would still be componentwise. F16/BF16 fused vs unfused intermediate rounding can differ. |
| GLU/SwiGLU, unary.cu (LOCAL) | Fixed block size; total count changes grid; row width determines gate addressing. | Same elementwise gate formula. Dense/expert matvec-fused GLU uses 0035v4/0008 singleton replay; preserving that epilogue is essential. |
| UNARY sigmoid/SiLU/softplus and fused UNARY+MUL (LOCAL) | Grid/count only. | Same scalar formula within a fusion; materializing an intermediate can change rounding. |
| SCALE, CLAMP, FILL (LOCAL) | Total element count changes grid, fixed threads. | Same scalar scale*x+bias, clamp, or constant store; no token reduction. |
| SOFT_MAX, softmax.cu:289/325 | Router width256 uses soft_max_f32<true,256,256>, independent of N. Grid only. | Same max/sum tree at width256. Shared-memory choice depends on width/device limit, not N. Supplied decode op inventory has no standalone SOFT_MAX; router may fuse. |
| Very-wide SOFT_MAX fallback, softmax.cu:356 | YES: cooperative launch when width/rows>8192, shared memory insufficient, cooperative support, no mask/sinks, scale1, bias0. | NO general guarantee: cooperative multi-CTA sum/division differs from per-row block sum/reciprocal-multiply. 0038 replays eligible rows through singleton dispatch. Not a width256 explanation. |
| Fused router/topk-moe.cu:139/340 | Fixed one warp per token and fixed rows per block; experts choose specialization. | Same warp reductions, selection/tie rule and normalization for the same fusion/configuration. Standalone softmax/ARGSORT is a different algorithm. |
| quantize_q8_1, quantize.cu:53/557 | Fixed threads, row count only changes grid. | Yes: each 32-value warp subgroup reduces max/sum within a padded row (ne0%32==0); same rounding and half2 scale/sum. |
| GET_ROWS, getrows.cu:234 | YES: same-type vector path requires total vector blocks>=128 plus alignment. | Same bits: vector/scalar gathers copy; converting/dequantizing paths use fixed element/block operations. No row-count-selected reduction. |
| SET_ROWS, set-rows.cu | Grid/count only; type chooses copy/conversion/block quantizer. | Same element conversion or fixed quantization block. Requires unique cache destinations; duplicate writes are races, not reduction-order equivalence. Fused ROPE scatter retains this condition. |
| ARGSORT/TOP_K | YES, CUB singleton/segmented algorithms differ; 0036v2 forces ARGSORT chunks=1 when GGML_CUDA_ARGSORT_EXACT_ROWS=1, and builds singleton indexer TOP_K nodes when LLAMA_EXACT_ATTN_COLS=1. | Same singleton dispatch with those flags after 0036, subject to existing singleton TOP_K boundary ties/nondeterminism. Exact-row ARGSORT has its own GGML_CUDA_ARGSORT_EXACT_ROWS flag. |
| CONCAT, CONT/DUP/CPY (LOCAL) | Shapes/contiguity affect grid and copy/transpose/vector/memcpy selection. | Same copy or per-element cast for the GLM float/half paths; no cross-row sum. Quantized copies remain block-local. |
| SUM_ROWS, sumrows.cu/reduce_rows.cuh (LOCAL) | YES: 512 threads when rows<2*SMs; else32 for width<1024, otherwise128. Both contiguous and strided paths. | Different tree in general. Router width8 rows=N<=8 stays512 on GB10; not a plausible fd6 trigger. 0038 forces singleton512 for all multi-row exact-mode calls. |
| CUMSUM (LOCAL; absent from decode inventory) | YES: CUB DeviceScan vs block scan depends on rows==1/width>1024 or width/rows>4096. | Different floating scan trees possible. No GLM decode CUMSUM use found; excluded from 0038, no universal exactness claim. |
| Weighted MoE merge (LOCAL) | Fixed256 threads, token only in grid/address. | Ordered expert fold stays the same for fixed used-expert order; production leaf missing. |
| LIGHTNING_INDEXER | Production leaf missing; LOCAL has tiled 4-head n_batch threshold and an existing limited spec-exact replay. | Matmul-like score reduction; cannot certify production dispatch. 0036 TOP_K changes do not certify this score kernel. |
| VIEW/RESHAPE/PERMUTE | Metadata only. | No arithmetic; downstream strides/eligibility can change. |

Fusion re-read: ggml-cuda.cu:4219 RMS matcher has type/layout/broadcast guards, no row-count block-size choice.
common.cuh:383 alignment checks base AND nb[1..3], including singleton axes; changed allocation/view alignment can select a different RMS tree without a row-count branch.
Router fusion memory-range guard at ggml-cuda.cu:3940 permits logits aliasing only when rows<=TOPK_MOE_ROWS_PER_BLOCK; crossing it can change fusion eligibility, not the warp arithmetic.
Other graph-use, layout, aliasing and concurrency guards can change the selected fusion. Digest traces cannot measure these production decisions.
Thus no supplied fused RMS N-dependent reduction explains fd6. Compare actual N=1/N=2 fusion selections, specialization/alignment and matvec singleton epilogues without the digest observer.
0038 retains every fused epilogue; it is a preventive correction of two proven row-count-sensitive standalone dispatch paths, not a demonstrated fd6 fix.

Validation: python3 tests/check-row-exactness.py (rowexact-test.log): full stack 0018→0010→0008→0034v2→0035v4→0036v2→0037→0038 apply/whitespace and 0038 reverse checks pass.
SUM_ROWS preimage is supplemented from LOCAL llama because production evidence omits it; its installed-source match remains to be checked on GB10.
Host C++ executes extracted patched dispatchers with launch spies: sum contiguous/strided, rows1/2/7/8/9/64 and widths8/256/2048; softmax rows1/2/8 on all three axes, widths256/8192/16384/65536, cooperative support and mask/sinks/scale/bias guards.
Flags 1/0/10/empty pass; singleton dispatch and per-row offsets checked. These tests validate host dispatch, not GPU numerical equality, capture/scratch lifetime, or speed.
Two careful reads completed: (1) kernel reductions, vectorization, indexing and fusion guards; (2) patch preconditions, singleton preservation, strided addressing, replay termination, source provenance and test coverage.
