# CUDA fusion classes (0035 v7)

`GGML_CUDA_DISABLE_FUSION_CLASSES` is an exact, case-sensitive comma list with no spaces. Empty/unset preserves v6. It is read once per process. Unknown names have no effect. `GGML_CUDA_DISABLE_FUSION=1` still overrides everything. These names are groups, including all listed variants and fallbacks within each group.

| Exact name | Fused operations |
| --- | --- |
| `matvec` | Dense/routed MUL_MAT[_ID] + ADD[_ID]; paired up/gate matmuls + optional NVFP4 scales and biases + GLU. Both float and quantized paths. |
| `shared_expert` | Joint routed ID up/gate/GLU and shared dense up/gate/GLU (six nodes). Separate from `matvec`; disabling both removes both classes. |
| `moe_weighted_reduction` | Optional expert-scale MUL, router-weight MUL, VIEW/ADD chain or compact expert reduction. |
| `topk_moe` | Sigmoid, softmax, or sqrt(softplus) router; reshape, optional bias/group selection, ARGSORT/VIEW/GET_ROWS, optional normalization/clamp/divide/scale; delayed-softmax variant. |
| `rms_norm` | RMS_NORM + MUL, optionally ADD. |
| `rms_norm_scale` | RMS_NORM + SCALE. |
| `rms_norm_rope` | RMS_NORM + MUL + ROPE, optionally VIEW + SET_ROWS. |
| `rope_set_rows` | ROPE + VIEW + SET_ROWS. |
| `multi_add` | Ordered chain of 2..8 ADDs. |
| `multi_mul` | Ordered chain of 2..8 MULs. |
| `unary_mul` | SiLU, SIGMOID, or SOFTPLUS + MUL. |
| `relu_sqr` | ReLU + SQR. |
| `softcap` | SCALE + TANH + SCALE. |
| `snake` | MUL + SIN + SQR + MUL + ADD. |
| `ssm_conv` | SSM_CONV + optional ADD + SiLU. |
| `gdn_cache` | GATED_DELTA_NET + state-snapshot CPYs. |
| `ssm_scan_cache` | SSM_SCAN + state-snapshot CPYs. |

Shared-expert, weighted-reduction, and topk switches guard their matchers, including allocator/graph optimization callers; other classes guard dispatch or `ggml_cuda_can_fuse`. No eligible class falls through to another variant of itself. A longer class may fall back to a different enabled class (e.g. `rms_norm_rope` to `rms_norm`), intentionally.

DeepSeek V4 HC COMB/PRE/POST are standalone GGML ops dispatched by `ggml_cuda_compute_forward`, outside `ggml_cuda_can_fuse` / `ggml_cuda_try_fuse`. The startup `resolve_fused_ops` text probes backend support, not proof that GLM executes HC. This switch does not change graph-level HC selection.

`win2fdb.sh` omits graph-absent rms_norm_scale, ssm_conv, gdn_cache, ssm_scan_cache, snake, relu_sqr, softcap, per the supplied GLM inventory. It retains uncertain ROPE patterns as candidates. Cold-only 128-token requests avoid prime/warm costs. There are 24 fixed server legs (two controls plus ten classes, each N=1/N=2). Typical ~110 seconds/server gives ~44 minutes. Adaptive pairs use remaining time; each is admitted only with 600 seconds remaining. Each server has a 300-second cap and the whole run a 54-minute watchdog, including an explicit DONE on budget expiry. Failed servers continue, but may exhaust the budget before every class/pair finishes; skipped pairs are named. It cannot guarantee exhaustive pairs under arbitrary failures and the fixed time limit.

GPU invocation on the fd9 host: stage the runner-case manifest files, then `bash ./win2fdb.sh > legs-fdb.txt 2>&1` from `/srv/glm-5.3`. The local sandbox has neither that tree/model/store nor CUDA tooling; no GPU results were generated here.

Candidate source review: supplied `norm.cu:399` uses width and alignment to choose vectorization/block size; row count only changes grid size. Supplied `binbcast.cu:169` collapses contiguous dimensions based on broadcast ratios, then `:301` derives multidimensional block geometry from the collapsed shape; at `:319` very large grids select an unravel kernel. It contains elementwise ordered folds, no cross-row reduction. Local `moe-weighted-reduction.cu` uses fixed 256-thread blocks and one token per grid-x, an ordered expert fold; the production evidence leaf is missing. Topk already replays singleton rows in v6. These observations do not establish the cause of the fd11 bytes.

`0035-v7-multi-binbcast-replay.candidate.patch` is an OPTIONAL follow-up against v7, deliberately not applied by the bisect window. Under EXACT_COLS and 2..8 total rows it slices each input (including broadcast operands) and output and calls the original fused ADD/MUL entry once per row. It retains the ordered fused chain. Host launch tests cover axes 1/2/3 and broadcast/varying operands; CUDA arithmetic, aliasing on production layouts, and whether it fixes the actual divergence remain unmeasured. Do not label it the culprit fix until the bisect identifies this class and GPU validation confirms equality. Applying it during the initial bisect would invalidate the requested control hashes.
