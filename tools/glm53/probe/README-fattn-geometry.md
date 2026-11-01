Run on the Spark from the directory containing `win2fag.sh` and `tools/`:

```bash
bash win2fag.sh
```

This uses `~/llama-k3-fd6/ggml/include` and existing shared libraries under
`~/llama-k3-fd6/build-fd4`. It compiles only the probe with
`CUDACXX=/usr/local/cuda/bin/nvcc -arch=sm_121`; it does not configure, patch,
or rebuild the production tree. No model, server, network, or scaling of
production is required. It consumes GPU resources while running, so concurrent
production activity can affect elapsed time (this probe makes no timing claims).
Keep the directory layout when transferring this bundle. Build and runtime logs
are preserved in a new `fageom-<UTC>-<pid>/` directory. Override `FAGEOM_TREE` or
`FAGEOM_OUT` if needed; an existing output directory fails rather than overwrites.

Direct build/link and run:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc bash tools/glm53/probe/build-fattn-geometry.sh "$HOME/llama-k3-fd6" ./fattn-geometry
./fattn-geometry
```

The build script discovers library directories and embeds their runtime paths;
link libraries are `-lggml -lggml-cuda -lggml-base`. A production build with the
public `ggml_flash_attn_ext_set_n_kv_max` and `ggml_prec_set_acc` APIs is required.
There is deliberately no CPU fallback.

`evidence/glm-op-profile.log` records GLM-5.3 head_count=64, head_count_kv=1,
kv_lora_rank=512, n_rot=64, unabsorbed MLA key dimension=256, freq_scale=1,
rope_yarn_log_mul=0. The absorbed FA inputs are Q F32 [576,1,64,1],
K F16 [576,E,1,1], and V=K[:512] with the same 576-element row stride.
The graph uses F32 accumulation, scale=1/sqrt(256)=1/16, no sinks, no ALiBi,
and no softcap, matching this model's singleton absorbed attention call.
The probe begins after Q projection/RoPE/cache storage; these frozen bytes
stand in for already positioned Q/K. It excludes output projection and indexer
score arithmetic to isolate attention geometry and TOP_K selection.

Each run uses the same counter-generated Q and cache prefixes, including the
same nonzero padded cache bytes. Indexer scores are unique finite values at
positions [0,n_valid), otherwise -INF. `ggml_top_k` + `ggml_cont` matches
`glm-dsa.cpp`; SET_ROWS and mask addition match `llama-graph.cpp`.
The base mask is F16 [E,1,1,1], 0 at valid positions and -INF elsewhere.
Both limits are tested: min(E,2048) and ceil(n_valid/256)*256. The latter is
an experimental K control, not the production policy. E=256,n_valid=300 is
impossible and explicitly skipped. All other combinations yield 34 runs.

Every line prints FNV-1a-64 over native output bytes, ranked TOP_K IDs,
sorted all-selected IDs (including padding), and sorted valid membership.
Sorted all-selected digests can differ solely because padded IDs/counts differ;
only sorted valid membership and finite-mask positions must remain identical.
The program checks those invariants and finite output, then reruns the same
graph and requires identical FA output. It compares full bytes, not only hashes,
to the smallest feasible E with production K for each n_valid, and prints max
absolute differences. FNV digests use native byte order (Spark is little endian).

Predictions follow the supplied production fattn.cu and fattn-mma-f16.cuh:
576/512, GQA64, singleton selects MMA F16 tile 1x16; sparse gather is selected
at E>=max(4096,2*K), otherwise dense. Effective KV partition length is K for
sparse and E for dense. No public API supplies a kernel launch trace; these are
explicitly predictions assuming this dispatcher and sm_121. Actual occupancy,
stream-k block count and fixup/reduction topology are not observed.

A GEOMETRY_SENSITIVE summary demonstrates changed FA arithmetic for frozen
inputs and identical membership, supporting Review 17's mechanism. It does not
prove this is the only production divergence. NO_DIFFERENCE_FOR_TESTED_INPUTS
means this seed/shape did not reproduce it, not a general extent-insensitivity
proof. Review 17's standalone-E fix (0036 v4) still needs production validation.
