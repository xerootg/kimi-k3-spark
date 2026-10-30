# GLM-5.3 expert kernels for the Intel iGPU and CPU (phase 1, laptop)

2026-10-29. Model `zai-org/GLM-5.3` (llama.cpp arch `glm-dsa`), Unsloth GGUF quants **UD-Q2_K_XL** (254 GB) and
**UD-Q4_K_XL** (467 GB), experts held in memory. Phase 1 ran only on the laptop (framework13: Core Ultra 7 165H,
Arc Meteor Lake iGPU `8086:7D55` on i915, 93 GB RAM). No production box was touched. Code: `tools/glm53/`.

## Summary

- **The quant is not what its name says.** UD-Q2_K_XL's routed experts are **IQ2_XS** (gate/up) and **IQ3_XXS** (down),
  not Q2_K/Q3_K. Q2_K/Q3_K appear only in the MTP layer (blk.78). UD-Q4_K_XL's experts are Q4_K (gate/up) and Q5_K
  (down). **The two quants share no routed-expert kernel type**; Q5_K/Q6_K are shared with Q2_K_XL's attention and
  shared expert only.
- **Upstream ggml-sycl already gets native DP4A.** With IGC 2.36.3, `dpct::dp4a` (byte extraction plus 4 multiplies) is
  pattern-matched to the hardware `dp4a` in every MoE vec-dot kernel for these types. The k3x problem was specific to
  `sycl::ext::oneapi::dot_acc` (on IGC 2.34.4). It is not a general speedup.
- **What limits the IQ kernels is the memory access pattern, not arithmetic.** IQ2_XS/IQ3_XXS super-blocks are 74/98
  bytes, so every field is only 2-byte aligned and a lane issues many small loads. Removing the grid lookups and sign
  work entirely changed IQ2_XS gate/up from 38 to 40 GB/s. Repacking each row so that every super-block field forms one
  aligned run (same bytes, no padding) took it to 64-70 GB/s.
- **g53x (standalone SYCL) against ggml-sycl: UD-Q2_K_XL experts 1.6-2.4x at k=1 and 2.1-3.0x at k=8; UD-Q4_K_XL
  1.4-1.7x at k=1 and 1.05-1.24x at k=8** (ggml-sycl's Q4_K/Q5_K MoE path is already near the roofline). k=8: 197-270
  us per expert (Q2) and 359-418 us (Q4), 61-66 GB/s end to end against a measured 75 GB/s iGPU read roofline (the
  kernels themselves run at 63-71 GB/s).
- **g53c (AVX2 + AVX-VNNI CPU) against ggml-cpu: 1.15-1.75x** (one case, Q2 L3 at k=1, 0.94x within noise), still
  compute-bound on IQ2_XS (22-33 GB/s at k=8 against 63 GB/s).
- **Correctness:** every g53x and g53c output equals the exact (fp64) sum over the kernel's own q8_1 inputs to 1e-7
  (g53c: 1e-6) relative, for all six weight types. Against an fp64 reference on the unquantized activations, the result
  is as accurate as ggml-sycl's (worst-row cosine 0.99992) and better than ggml-cpu's q8_K path (0.99978).

## 1. Quant inventory (HTTP Range reads of the GGUF headers only)

`gguf_hdr.py` reads each shard's header with Range requests; `inventory.py` classifies the tensors. Full output:
`tools/glm53/results/inventory-*.txt`.

Architecture: 79 blocks = 3 dense + 75 MoE + 1 MTP (blk.78, `nextn_predict_layers = 1`), 256 routed experts, 8 used,
1 shared expert, hidden size 6144, expert FFN 2048, dense FFN 12288, MLA attention (q_lora 2048, kv_lora 512) plus the
DSA indexer (32 heads x 128, top-k 2048), vocabulary 154880.

| class | UD-Q2_K_XL types | GB | UD-Q4_K_XL types | GB |
|---|---|---|---|---|
| routed gate / up (75 layers) | IQ2_XS (74 layers), IQ3_XXS (L8) | 70.1 + 70.1 | Q4_K (74), Q5_K (L8) | 136.3 + 136.3 |
| routed down | IQ3_XXS (71), IQ4_XS (L8, L75-77) | 94.4 | Q5_K (71), Q6_K (L8, L75-77) | 167.8 |
| shared expert gate / up / down | Q5_K / Q5_K / Q6_K | 2.08 | Q8_0 | 3.00 |
| dense FFN (L0-2) | Q5_K / Q6_K | 0.51 | Q8_0 | 0.72 |
| attention output, q_a | Q5_K | 6.09 | Q8_0 | 9.38 |
| attention q_b, kv_a, k_b, v_b | Q8_0 | 4.29 | Q8_0 | 4.29 |
| DSA indexer | Q8_0 + F32 | 0.82 | Q8_0 + F32 | 0.82 |
| router (F32) | | 0.47 | | 0.47 |
| output head | Q4_K | 0.54 | Q8_0 | 1.01 |
| token embedding (1 row per token) | Q5_K | 0.65 | Q8_0 | 1.01 |
| MTP layer (blk.78) | Q8_0, Q5_K, Q2_K, Q3_K, Q6_K | 3.76 | Q8_0, Q4_K, Q5_K | 6.15 |
| **total** | | **253.9** | | **467.3** |

| per token (decode, no MTP) | UD-Q2_K_XL | UD-Q4_K_XL |
|---|---|---|
| one routed expert (gate + up + down) | 12.22 MB average (12.09-16.32) | 22.94 MB (22.81-27.62) |
| routed, 8 x 75 experts | 7.33 GB | 13.76 GB |
| everything else read every token (attention incl. indexer, shared expert, dense FFN, router, output) | 14.79 GB | 19.72 GB |
| **total read per token** | **22.1 GB** | **33.5 GB** |
| all 19200 routed experts | 234.7 GB | 440.4 GB |
| experts in 64 GB (nuc2-class, raw) | 5236 (27.3 %) | 2790 (14.5 %) |
| experts in 32 GB (nuc3-class, raw) | 2618 (13.6 %) | 1395 (7.3 %) |
| experts in ~48 GB / ~24 GB usable (what nuc2 gives K3 today / a guess for nuc3) | 3928 / 1964 | 2092 / 1046 |

For phase 2: **UD-Q5_K_XL** is 562.5 GB (gate/up Q5_K, down Q6_K, Q8_0 down in 4 layers; 27.83 MB per expert, 36.4 GB
read per token). It does not fit in the GH200's ~537 GB of HBM + LPDDR, so it needs the expert store there too.

## 2. Test set

`extract.py` pulls experts 0-7 of the gate/up/down 3-D tensors with Range requests (an expert is a contiguous slice
of `ne0 x ne1` elements), for one layer of each distinct type combination: UD-Q2_K_XL L3 (IQ2_XS / IQ3_XXS), L8
(IQ3_XXS / IQ4_XS), L75 (IQ2_XS / IQ4_XS); UD-Q4_K_XL L3 (Q4_K / Q5_K), L8 (Q5_K / Q6_K), L75 (Q4_K / Q6_K). 895 MB in
the scratch area, not committed. The activation is a seeded N(0,1) vector (the same in every tool), not a real
hidden state; real routing ids are not needed for kernel timing.

## 3. Baseline (ggml on the laptop)

`glmbench.cpp` is `tools/distributed/moebench.cpp` adapted to GLM: one layer's routed FFN for one token (MUL_MAT_ID
gate and up, SwiGLU, MUL_MAT_ID down, weighted sum) on a ggml backend. The 8 real experts are replicated into 64
slots and each iteration draws k distinct slots, so the weights stream from DRAM. Tree: the fork's `kimi-k3` branch
(aa057715c), SYCL built in `intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04` (its compute-runtime 26.22 / IGC
2.36.3; the NUCs' pinned runtime is 26.18 / IGC 2.34.4), CPU built natively.

Rooflines measured on this laptop: iGPU streaming read **75 GB/s** (`bw.cpp`), CPU **63 GB/s** at 16-22 threads
(`bwcpu.c`). The SMBIOS table is root-only; 93 GB RAM on a 165H means 2 x 48 GB DDR5-5600 SO-DIMMs, so the DRAM peak is
89.6 GB/s (inferred, not read). The laptop was in use during all runs (load 3-4), so p50s move by ~10-20 % between runs.

The ggml-sycl numbers decompose into a large fixed cost and a per-expert slope: UD-Q4_K_XL L3 ~530 us fixed plus 302 us
per expert (76 GB/s, at the roofline); UD-Q2_K_XL L3 ~630 us fixed plus 458 us per expert (27 GB/s).

**The DP4A check.** IGC assembly dumped with `IGC_ShaderDumpEnable=1` and counted with `asmstat.sh`: ggml-sycl's
`mul_mat_vec_q_moe` kernels contain native `dp4a` (IQ2_XS 414 instructions / 8 dp4a / 13 sends, IQ3_XXS 454 / 8 / 17,
IQ4_XS 680 / 8 / 43, Q4_K 371 / 4 / 14, Q5_K 400 / 4 / 16, Q6_K 456 / 4 / 19 per loop body). `dpct::dp4a` is the scalar
form, and IGC recognises it; k3x's emulated-multiply problem came from `dot_acc`. Not rechecked on IGC 2.34.4; g53x
calls `__builtin_IB_dp4a_ss` directly, which k3x proved native on 2.34.4.

## 4. Optimized kernels

### g53x: iGPU, standalone SYCL (`g53x.cpp`, `g53x_ty.inc`)

Two launches per token: gate+up+SwiGLU (q8_1 of x inside the kernel), then down (q8_1 of h inside the kernel).
256-item work-groups, one sub-group of 16 lanes per 2 gate rows and their 2 up rows (down: 2 rows). The lanes stride
over the row's 32-value units, and each unit's activation is loaded from SLM once for all rows of the sub-group.

What each step bought (UD-Q2_K_XL L3, k=8, kernel GB/s, gate+up / down):

| step | gate+up | down |
|---|---|---|
| ggml-sycl baseline (whole token) | ~20 | |
| ggml block layout, grid + sign tables in SLM, 1 row per sub-group | 30 | 25 |
| sign masks in ALU (popcount + shifts) instead of SLM, 2-4 rows per sub-group sharing the activation load | 34-38 | 27-31 |
| (experiment: grid and sign work removed, results wrong) | 40 | |
| **rows repacked into per-field runs** (qs, scales, d each contiguous and aligned; same size) | **64-66** | **69-71** |
| q8_1 fused into both kernels (bit-identical output) | ~same | ~same |
| immediate command lists (`UR_L0_USE_IMMEDIATE_COMMANDLISTS=1`, set by `main`) | whole token at k=1: 607 -> 337 us in one A/B (noisy) | |

The K-quants needed one change for exactness: the q8_1 block sum used by the Q4_K/Q5_K min term is `d * sum(q)`, not
ggml's float `sum(x)`. With the float sum the result differs from the exact quantized dot by 0.5 %. Q6_K was first
written with units spanning 4 activation blocks (8 scattered SLM loads per unit: 30-50 GB/s); one 32-block per unit
fixed it (69-71 GB/s).

Results (`results/igpu-mtl-165h.txt`), p50 us per token, ggml-sycl -> g53x (GB/s of expert weights):

| slice | k=1 | k=2 | k=4 | k=8 | g53x us/expert at k=8 |
|---|---|---|---|---|---|
| Q2_K_XL L3 IQ2_XS/IQ3_XXS | 914 -> 560 | 1522 -> 496 | 2561 -> 851 | 4777 -> 1574 (20 -> 62) | 197 |
| Q2_K_XL L8 IQ3_XXS/IQ4_XS | 891 -> 389 | 1403 -> 639 | 2335 -> 1127 | 4444 -> 2158 (29 -> 61) | 270 |
| Q2_K_XL L75 IQ2_XS/IQ4_XS | 886 -> 372 | 1070 -> 574 | 1935 -> 998 | 3621 -> 1746 (31 -> 64) | 218 |
| Q4_K_XL L3 Q4_K/Q5_K | 882 -> 509 | 871 -> 804 | 1573 -> 1533 | 3017 -> 2871 (61 -> 64) | 359 |
| Q4_K_XL L8 Q5_K/Q6_K | 738 -> 517 | 1174 -> 938 | 2124 -> 1714 | 4136 -> 3345 (53 -> 66) | 418 |
| Q4_K_XL L75 Q4_K/Q6_K | 889 -> 519 | 1080 -> 888 | 1927 -> 1597 | 3710 -> 3073 (53 -> 64) | 384 |

At k=1 the GPU kernels take ~250-300 us and the rest is submission, wake-up and clock ramp (the k3x RC6 and memory-clock
lessons apply; forcewake and memwarm were not used here).

### g53c: CPU, AVX2 + AVX-VNNI (`g53c.cpp`)

The same q8_1 activations as g53x, ggml's blocks unchanged, `vpdpbusd` for every dot (IQ: grid bytes are unsigned, the
signs go onto the activations with `vpsignb`; IQ4_XS: table values + 128 and a per-block correction; Q4_K/Q5_K:
nibbles + high bits). A spinning pool; a gate/up task is 32 rows and also quantizes the h block it produced, so down
starts after one barrier. Gathers (`vpgatherqq`/`vpgatherdd`) for the IQ tables were slower than scalar loads here. No
Q6_K yet.

16 threads, p50 us per token, ggml-cpu -> g53c (`results/cpu-mtl-165h.txt`):

| slice | k=1 | k=2 | k=4 | k=8 (GB/s) |
|---|---|---|---|---|
| Q2_K_XL L3 | 914 -> 972 | 1484 -> 1305 | 2731 -> 2276 | 5866 -> 4429 (16.5 -> 21.8) |
| Q2_K_XL L8 | 1637 -> 989 | 1981 -> 1254 | 3264 -> 2210 | 6755 -> 4007 (19.3 -> 32.6) |
| Q2_K_XL L75 | 1400 -> 867 | 1882 -> 1491 | 3122 -> 2156 | 5700 -> 4102 (19.6 -> 27.2) |
| Q4_K_XL L3 | 1211 -> 693 | 1268 -> 940 | 2282 -> 1750 | 4071 -> 3554 (44.8 -> 51.3) |

IQ2_XS on the CPU stays compute-bound (four table loads plus inserts per 32 weights). On this laptop the iGPU is 2-3x
faster per expert than the CPU for GLM's experts.

## 5. Correctness (`check.sh`, `glmref.cpp`, output in `results/check.txt`)

For each slice: ids 0..7, the same x, through ggml-cpu, ggml-sycl, g53x and g53c, then `glmref`:
- **Exact arithmetic:** g/u/down rows against the fp64 sum over the weights (dequantized with ggml's `to_float`) and the
  kernel's own q8_1 activations. g53x: max error / max value 1.0-2.2e-7 for every type (float rounding of the final
  sums; the integer parts are exact). g53c: 2e-7 to 1.2e-6.
- **Against fp64 on the unquantized x:** worst-row cosine g53x 0.99992 (= ggml-sycl 0.99992), g53c identical to g53x,
  ggml-cpu 0.99978 (its q8_K activations use one scale per 256 values).
- **Between implementations:** g53x vs ggml-sycl max relative 1.3-2.8e-3, g53x vs g53c 2e-7 to 2e-4 (a rounding flip in
  q8_1(h) moves a whole 32-block).

One harness bug found on the way: ggml's graph allocator reused the down output's buffer in place for the weighting
multiply, so `--check` read rows scaled by 1/k; the check now marks those tensors as outputs.

## 6. What transfers

- **NUCs (Arrow Lake H, Arc 130T, nuc2 225H / 64 GB):** the same Xe-LPG-class iGPU and DP4A, one Xe core fewer than the
  laptop, DDR5 at a similar or higher peak. Expect per-expert times within ~15 % of the table above; this must be
  measured on an Arrow Lake box, which needs a window when K3 does not need nuc2/nuc3. The k3x serving lessons (host-USM
  arena, immediate lists, forcewake, memwarm, EPP performance) carry over unchanged, and the field-run repack fits the
  k3x export step (records are repacked once at load).
- **Spark (GB10) and GH200 (CUDA):** the main lesson is the layout. ggml CUDA's IQ vec-dots read the same 74/98-byte
  blocks with 2-byte-aligned `get_int_b2` loads, so the per-field repack is worth testing there (it is the same idea as
  the Spark's aligned expert store). GLM's UD-Q5_K_XL experts are Q5_K/Q6_K: the Q6_K TMA GEMV work (roadmap step 4,
  `tools/q6k-gemv-bench.cu`) applies directly to its down projections. Bit-exactness against a CUDA build is a later
  step: it needs the CUDA reduction order and CUDA's float `sum(x)` in the q8_1 block sum, which g53x replaced with
  `d * sum(q)` (one line to switch back).
- **llama.cpp upstream:** ggml-sycl already repacks Q4_K/Q5_K/Q6_K/Q8_0 for its MoE path ("reorder"). Adding IQ2_XS /
  IQ3_XXS / IQ4_XS to that reorder with this field-run layout should give the 2.5-3x seen here for any IQ MoE model on
  Intel iGPUs. A good upstream contribution.

## 7. UD-Q2_K_XL against UD-Q4_K_XL for a NUC expert pool

| | UD-Q2_K_XL | UD-Q4_K_XL |
|---|---|---|
| bytes per expert | 12.2 MB | 22.9 MB |
| g53x per expert, k=8 (laptop) | 197-270 us | 359-418 us |
| g53x token, k=1 / k=2 (laptop) | 370-560 / 500-640 us | 510-520 / 800-940 us |
| experts per 64 GB box (raw / ~48 GB usable) | 5236 / 3928 | 2790 / 2092 |
| experts per 32 GB box (raw / ~24 GB usable) | 2618 / 1964 | 1395 / 1046 |
| Spark expert cache, if GLM replaced K3 there (~85 / 80 GB) | ~6950 experts (36 %) | ~3490 (18 %) |
| share of the Spark's misses removed by nuc2 / nuc3 / both: uniform-routing bound | 32 % / 16 % / 48 % | 13 % / 7 % / 20 % |
| same, scaled by K3's measured skew (a remote set removes ~0.4x its uniform share, `docs/experiments.md` routing replay) | ~13 % / ~6 % / ~19 % | ~5 % / ~3 % / ~8 % |

The miss shares are estimates, not measurements: GLM routing traces do not exist yet. They assume the Spark's GLM
cache takes the memory K3's spine frees (GLM's non-expert weights are 14.8-19.7 GB against K3's 45.5 GB). Per byte the
two quants cost the same iGPU time; Q2_K_XL fits 1.9x more experts per box and halves the time per remote expert.
Q4_K_XL is the quality choice, and the Spark would stream almost twice the bytes per token for it (33.5 vs 22.1 GB).

## 8. Phase 2 plan

1. **GLM routing traces.** Run UD-Q2_K_XL with the expert-id dump (`k3-pfdump` equivalent) for prose and code, then
   `remotesim.py` / `remoteset.py` for GLM's 75 x 256 experts, to replace the estimates in section 7 with measured miss
   shares and pick each box's set. This needs a host with ~254 GB (the GH200, or the Spark streaming from NVMe) during a
   K3 window: the user's call.
2. **g53x serve mode:** the k3x wire protocol and arena (`serveidx`, host-USM records, immediate lists, forcewake,
   memwarm, trace) with hidden size 6144, types per record, and the field-run repack done at export. Then the CPU
   split from k3x (`K3X_CPU_SPLIT`), with g53c as the CPU half and a Q6_K CPU kernel.
3. **Measure on Arrow Lake** (nuc2-class) under the k3x conditions (EPP performance, forcewake, memwarm, 8 ms gaps),
   k = 1-3 like the engine.
4. **Upstream ggml-sycl:** the IQ2_XS / IQ3_XXS / IQ4_XS reorder with the field-run layout, measured with `glmbench`.
5. **GH200, UD-Q5_K_XL:** 562 GB > ~537 GB of HBM + LPDDR, so it runs on the expert store with ~95 % resident. Its experts
   are Q5_K/Q6_K: test ggml CUDA's GEMV against the Q6_K TMA GEMV and the field-run layout, when K3 can spare the box.

## Files (`tools/glm53/`)

- `gguf_hdr.py URL OUT.json` (one shard's header via Range), `inventory.py HDRDIR` (classes, bytes per token, fit),
  `extract.py HDRDIR QUANT LAYERS OUTDIR` (experts 0-7 of the given layers).
- `glmbench.cpp` + `CMakeLists.txt`: ggml baseline and `--check` dumps. `glmref.cpp`: fp64 reference and comparisons.
- `g53x.cpp` + `g53x_ty.inc`: the iGPU kernels (`bench`, `check`; `G53X_PROF=1` per-kernel times). `bw.cpp`: iGPU
  read roofline.
- `g53c.cpp`: the CPU kernels. `bwcpu.c`: CPU read roofline.
- `build-cpu.sh`, `build-sycl.sh`, `dsycl.sh` (run in the oneAPI container with `/dev/dri`), `all.sh gpu|cpu`
  (tables above), `check.sh` (section 5), `asmstat.sh DUMPDIR REGEX` (instruction mix of IGC `.asm` dumps).
  Work dir `$W` (default `~/glm53-work`): `llama/` = the fork's tree, `data/` = `extract.py` output.
- `results/`: inventories, the two benchmark runs, the check output.
