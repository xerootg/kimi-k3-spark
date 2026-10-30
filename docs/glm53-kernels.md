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
   split from k3x (`K3X_CPU_SPLIT`), with g53c as the CPU half and a Q6_K CPU kernel. **Done (first part), see
   section 9:** the repack runs once at load instead of at export; no trace, no CPU split yet.
3. **Measure on Arrow Lake** (nuc2-class) under the k3x conditions (EPP performance, forcewake, memwarm, 8 ms gaps),
   k = 1-3 like the engine.
4. **Upstream ggml-sycl:** the IQ2_XS / IQ3_XXS / IQ4_XS reorder with the field-run layout, measured with `glmbench`.
5. **GH200, UD-Q5_K_XL:** 562 GB > ~537 GB of HBM + LPDDR, so it runs on the expert store with ~95 % resident. Its experts
   are Q5_K/Q6_K: test ggml CUDA's GEMV against the Q6_K TMA GEMV and the field-run layout, when K3 can spare the box.

## 9. g53x serveidx: GLM experts behind the k3x protocol

`g53x SET STORE HIDDEN - serveidx PORT` serves a K3XS expert store (`tools/repack-experts.py`, v1 or v2) with the
wire protocol of k3x `serveidx`, so the Spark engine (`rrclient.c`, patch 0003) talks to either server unchanged:

- Request: 17 int32 `{k, gid_1..gid_k, 0.., seq in hdr[16] when k < 16}`, then `HIDDEN` float32 x. Reply: k rows of
  `HIDDEN` float32 in request order, each streamed when its down kernel is done (`K3X_NOSTREAM=1`: one down launch,
  then all rows). If k is outside 1..16, the connection is closed. `gid = layer * n_expert + expert`, where n_expert
  comes from the store header (256 for GLM, 896 for K3).
- As in k3x: if any id is not served, k rows of NaN come back. That covers a layer not in the store, an expert not in
  SET, a layer with unsupported types (GLM's MTP layer 78, Q2_K/Q3_K), and ids of two kinds mixed in one request. The
  engine then computes those experts itself. With the disk tier (section 10) only the unservable row is NaN, as in
  k3cs. An all-zero row is sent as NaN and counted (`conn N: M all-zero rows sent
  as NaN so far` at connection close, as in k3cs). The log lines (`loaded ... host USM arena ...`, `self-check ok`,
  `serving on PORT (busy-poll)`, `K3X_MAX_GB ... reached`) are k3x's.
- The store is parsed by `tools/bitexact/expert_disk.h`, which now also reads v2 (per-layer stride). The tensor types
  and the FFN size of each layer come from its index entry: FFN = gate bytes / gate row bytes, and the up and down sizes
  are checked against it. The index stores only bytes, so the hidden size comes from `STORE.json`, the sidecar that
  repack-experts.py writes (`HIDDEN` = `-`; the gate tensor's `dims`), or from a number. Kernels are instantiated per
  (hidden, FFN) pair (`G53X_DIMS`, default 6144 x 2048): with a runtime row length the kernels ran about 1.5x slower.
- SET: `layer expert` lines (the engine's `LLAMA_EXPERT_STORE_REMOTE_SET`), k3x IDX lines `gid tg td offset` (checked
  against the store), or `all`. At startup every listed expert is read once with pread, repacked to the field-run
  layout, and placed in 1 GiB host-USM chunks. The request slot (header, x, rows, pointer table, h) is allocated before
  serving, so the request path allocates nothing. The kernels take a per-expert pointer table instead of slot ids.
- Env (k3x names): `K3X_MAX_GB` (24), `K3X_SETOUT`, `K3X_KEEPCACHE`, `K3X_NOSTREAM`, `K3X_RT`, `K3X_RX_SPIN_US`,
  `K3X_MEMWARM`, `K3X_MEMWARM_CPU`, `K3X_FORCEWAKE`. K3-only parts (SiTU, softcap, MUFU tables) do not apply. Trace,
  `K3X_CTL` and the CPU split are not ported yet. Disk tier (`K3X_DISK_*`) and the g53x-only `G53X_*` knobs: section 10.

Test: `./tools/glm53/build-sycl.sh && python3 tools/glm53/serve_test.py`. The test uses only the standard library. It
builds four 0.3-0.6 GB stores in `$W/tmp/serve_test` from `data/`: UD-Q2_K_XL and UD-Q4_K_XL, each as v1 and as v2
with `--stride-align 524288`. Each store holds layers 3, 8 and 75 (experts 0-7) plus a zero Q2_K/Q3_K layer 78. In the
container it then runs `g53x ... check` for the six slices and starts one server per store. The client asserts that
every served row is bit-identical to the check's `.dn` row for the same expert and x, at k = 1, 4, 7, 8 and 16
(repeats), in shuffled orders. It also checks the NaN cases above, that k = 0 closes the connection, that reconnecting
works, and that `K3X_SETOUT` lists the served set. Each store is checked streamed, and the v1 stores also with
`K3X_NOSTREAM=1`.

Per-request latency on this laptop (Core Ultra / Arc iGPU, the shared host at load average 5-6), measured by the client
from send to the last reply byte over loopback, median of 200, k = 1/4/8 in µs. L75 has 7 served experts, so its k = 8
repeats one.

| quant | store | L3 | L8 | L75 |
|---|---|---|---|---|
| UD-Q2_K_XL | v1 | 591 / 1271 / 2400 | 502 / 1554 / 3899 | 867 / 2854 / 5026 |
| UD-Q2_K_XL | v2 | 865 / 1801 / 4288 | 1074 / 2743 / 5515 | 1058 / 2779 / 4905 |
| UD-Q4_K_XL | v1 | 680 / 2204 / 5244 | 1157 / 3787 / 7476 | 1517 / 3281 / 6166 |
| UD-Q4_K_XL | v2 | 898 / 2363 / 5237 | 1278 / 3366 / 7488 | 1461 / 3320 / 9048 |

`g53x bench` in the same session (weights in device USM, no socket) gave Q2 L3 350/931/1741, Q4 L8 588/2135/6519 µs.
In a quieter minute it gave Q4 L8 534/1745/3382, while serveidx gave Q4 L8 655/2162/4092. Run-to-run spread on this
shared host is up to 2x. In the bench, host-USM weights (`G53X_HOSTW=1`) cost the same as device USM. The store version
does not change the arena: v1 and v2 differ only in where the records sit in the file. Measure the serve overhead
proper on an idle Arrow Lake NUC (item 3 of section 8).

## 10. g53x serveidx disk tier: any expert, streamed from the local NVMe

`K3X_DISK_TIER=1` lets serveidx serve every record of the store, not only the SET loaded at startup. With the variable
unset, the behaviour and the wire bytes are the same as in section 9: serve_test runs the old configurations unchanged.

- **Store, LRU, reads.** The store comes from `tools/bitexact/expert_disk.h`, shared with k3cs. `k3disk::Cache` reads
  with O_DIRECT into one preallocated arena of fixed slots, one per record, sized to the largest record. g53x passes an
  arena made with `sycl::malloc_host`, allocated once at startup, made resident with `zeContextMakeMemoryResident`
  and covered by `mlockall`, so the kernels read the slots zero-copy. The reads run on reader threads: `K3X_DISK_THREADS`
  is the number in flight, i.e. the queue depth; the container has no liburing, so the readers use pread. Each record is
  split into `K3X_DISK_PIECE_KB` pieces, and LRU eviction skips held slots. Env (k3cs names): `K3X_DISK_STORE` (default:
  the served STORE), `K3X_DISK_LRU_GB` (1), `K3X_DISK_DEADLINE_MS` (20), `K3X_DISK_THREADS` (8), `K3X_DISK_PIECE_KB` (512).
- **Repack.** The new `Options::transform` hook converts the field-run layout in place on the reader thread, as soon as
  a piece's rows have fully landed: rows that span two pieces are converted by whichever piece completes second. One
  row at a time goes through a 16 KiB stack buffer, so there is no heap buffer and no second pass. A record is ready
  after its last transform. k3cs passes no transform, so its behaviour is unchanged.
- **Overlap.** Once the 68-byte header arrives, the ids are classified and every miss's read is started
  (`acquire`); x arrives meanwhile. The resident rows go through gate/up and down first. The send loop then polls:
  each disk row is launched (gate/up + down into its own row slot) as soon as `settled()` says its record is complete,
  so each expert finishes on its own with no shared barrier. Rows are sent in request order, each as soon as its kernel
  is done.
- **Deadline.** It is measured from the header. A record that is not ready by then (`wait(handle, deadline)` returns
  null), or whose read failed, is answered with a NaN row, as an unserved expert would be, and the Spark computes it.
  The read itself keeps going and ends up in the LRU.
- **Stats** are printed on the connection-close line, as in k3cs: `disk: hits lru_hits bytes p50_us p99_us
  deadline_misses busy_ms gbps`. p50/p99 come from power-of-two latency buckets, and gbps = bytes ÷ time with at
  least one read in flight.

### Allocation and page-fault gate (`disk_test.py gate`, `allocgate.c`)

`allocgate.so` is an LD_PRELOAD interposer that counts, between two SIGUSR2 signals:
- malloc, calloc, realloc, posix_memalign/aligned_alloc/memalign, operator new and free;
- mmap and munmap;
- zeMemAllocHost/Device/Shared, zeMemFree, zeEventCreate, zeEventPoolCreate and zeCommandListCreate(Immediate).

The ze* calls can be interposed because SYCL's Level Zero adapter reaches them through the dynamic symbol table.

The gate (with ALLOCGATE_BT=n, allocgate also prints backtraces of the first n window allocations):
1. builds a 22.5 GB synthetic v2 store: layers 3-9 x 256 experts, each record a copy of one of the UD-Q2_K_XL
   layer-3 experts, 12.1 MB like GLM-5.3's (1792 records);
2. starts serveidx with resident experts 0-7 per layer and a 2 GB LRU;
3. warms up with 300 requests on one connection, then sends 50 more on a second;
4. opens the window and sends 1200 requests of k = 1..8, mixing resident, LRU-hit, cold and NaN rows;
5. asserts zero counts, zero minor and major faults (/proc/PID/stat fields 10 and 12), bit-exact rows against
   `g53x check` of the source experts, and disk stats equal to the client's own cold/LRU counts.

With `--fault-addrs N` (default 20), allocgate's perf_event_open sampler prints the address and maps line of each
fault; for that, the container runs with `seccomp=unconfined`.

Found on the request path, in order:
1. SYCL `submit`: a handler and a UR event per launch (~10 allocations). Fix: the kernels are SYCL functors,
   pre-recorded into Level Zero regular command lists at startup (one per kernel x batch size and per row slot,
   pointer-table arguments in USM), each executed once before serving. A list's first execution allocates.
   `G53X_SYCL_LAUNCH=1` restores the SYCL path.
2. NEO gathered every USM allocation for indirect-access kernels into a new vector on each execute (~4 `new`).
   - `MakeIndirectAllocationsResidentAsPack=1` avoids that, but under memory pressure it returned all-zero rows
     after 8-500 requests: the buffers had been evicted.
   - Fix: `DisableIndirectAccess=1` plus `zeContextMakeMemoryResident` on every USM buffer (arena chunks, LRU arena,
     request slot, pointer table).
3. i915 execbuffer pushed onto the GEM close worker's std::deque (512 B every 64 submissions). Fix: `EnableGemCloseWorker=0`.
4. The immediate command list grew its command-buffer vector each time its stream filled (one `new`). Fix: an in-order
   command queue with `zeCommandQueueExecuteCommandLists` (`G53X_SUBMIT=queue`, default; `immediate` = the old path).
5. Page faults:
   - Library and anonymous pages paged out under host memory pressure. Fix: `mlockall(MCL_CURRENT)` before serving
     (`G53X_MLOCKALL=0` turns it off; dsycl.sh raises the memlock limit).
   - NEO patched a jump-back into each regular list's batch buffer, through an i915 GEM CPU map, on every execute
     (~260 faults per window). Fix: `DispatchCmdlistCmdBufferPrimary=0`.
6. ULLS (`EnableDirectSubmission=1`) also removed 3 and 4, but on this i915 laptop it stalled the GPU now and then: the
   append spins forever. It is opt-in (`G53X_ULLS=1`). The ev_wait timeout (`G53X_GPU_TIMEOUT_MS`, 10 s) exits with
   status 4 instead of hanging.

The keys in 2, 3 and 5 are NEO debug keys, set by g53x itself (`NEOReadDebugKeys=1`; `G53X_NEO_TUNE=0` = the driver's
defaults).

Result on this laptop (93 GB RAM, swap 100 % full from other jobs):
- 0 allocations of any kind in the window (the ze totals are all at startup), every row bit-exact, stats matching;
- but 1-3 minor faults per 1200 requests, which fails the gate. Every one is on an `anon_inode:i915.gem` mapping: the
  command queue's 192 KiB ring buffer (the same IP in NEO each time) and once a 4 KiB BO. i915 GEM mmaps are PFN maps
  that mlock does not pin, and the shrinker zaps idle ones under memory pressure. Runs on a less loaded host gave 0/0.
- Check it on the NUC (xe KMD there, and the NEO keys need `NEOReadDebugKeys` support in its driver).

Exact modes (`EXACT-RESULT.md`, `FASTEXACT-RESULT.md`): `G53X_EXACT=2` (the default) is the fast CUDA-exact path
(`g53x_parallel.inc`), `G53X_EXACT=1` the scalar exact oracle; both give the same bits. The exact kernels are functors
like the fast ones (`ExactGateUpK` / `ExactDownK` in `g53x_exact.inc`; `ParPrepK` + `ParMatK`, the MUFU tables and IQ
grids as direct arguments), so serveidx pre-records them the same way and resident and disk-tier rows take the same
allocation-free path. Mode 2's lists are two kernels each (q8_1 preparation into a scratch slot, a barrier, the SIMD16
matvec); the scratch (one region per slot plus one for x) is reserved and made resident at startup, as are the 2.1 GB of
MUFU tables. Gate after the fast-exact merge: 0 allocations in modes 0 and 2; minor faults on i915 GEM mappings only,
36 (mode 0) and 31-74 (mode 2) with swap 100 % full, earlier 1-5 on a quieter host (`MERGE-RESULT.md`).

### Throughput (this laptop, `disk_test.py bench`)

Method:
- Store: the same synthetic 22.5 GB store, 2 GB LRU, deadline 1 s so latency is measured rather than NaN rows.
- 200 requests per cell; client-side send to last byte over loopback.
- Cold rows are never in the LRU; warm rows are resident.
- "Wall GB/s" = bytes ÷ the cell's wall time; busy = bytes ÷ time with a read in flight.
- The host was loaded (load average 7-9) and noisy; cells varied by up to 1.5x between passes.
- All cells: 0 NaN rows, 0 wrong rows.

Median / p99 µs, wall GB/s, at the default (512 KiB pieces, 8 readers) and 16 readers:

| k | cold % | 512 KiB x 8 | 512 KiB x 16 |
|---|---|---|---|
| 1 | 0 | 683 / 1233 | 763 / 1457 |
| 1 | 50 | 2848 / 5359, 2.4 | 1994 / 4388, 2.8 |
| 1 | 100 | 3808 / 4887, 3.2 | 3296 / 4481, 3.7 |
| 4 | 0 | 2194 / 3098 | 1647 / 2498 |
| 4 | 50 | 6493 / 11502, 3.6 | 6222 / 7270, 4.0 |
| 4 | 100 | 8383 / 9817, 6.0 | 6029 / 8646, 8.0 |
| 8 | 0 | 5279 / 6046 | 5292 / 6818 |
| 8 | 50 | 9656 / 11383, 5.3 | 10430 / 13182, 4.9 |
| 8 | 100 | 15572 / 17830, 6.5 | 16613 / 20672, 5.9 |

Read size x readers at k = 8, 100 % cold (median / p99 µs, wall GB/s):

| piece | 2 readers | 4 | 8 | 16 |
|---|---|---|---|---|
| 256 KiB | | | 15453 / 17406, 6.5 | 14809 / 18445, 6.7 |
| 512 KiB | 52506 / 85930, 1.9 | 44065 / 63680, 2.3 | 15572 / 17830, 6.5 | 16613 / 20672, 5.9 |
| 1 MiB | 34144 / 42821, 2.9 | 23048 / 35456, 4.1 | 20039 / 26135, 4.9 | 19538 / 38655, 4.8 |
| 4 MiB | 102945 / 144768, 1.0 | 74883 / 119313, 1.3 | 23760 / 46660, 3.9 | 24638 / 54786, 3.4 |

Findings:
- Many reads in flight matter more than their size. 4 MiB pieces leave only three reads per 12 MB expert, so a k = 1
  cold read cannot use the drive's parallelism.
- 256-512 KiB with 8-16 readers reached 6-8 GB/s wall, at the laptop drive's limit.
- The k = 8, 100 % cold case is bound by the drive (8 x 12.1 MB = 97 MB per request).

Defaults: 512 KiB pieces, 8 readers (as good as 16 at k = 8, half the threads), 20 ms deadline, 1 GB LRU.

On a NUC (TM8FGM002T at ~5.7 GB/s), one cold expert takes ~2.1 ms of reading alone, so a cold row fits the Spark's
3 ms budget only if the request has slack. LRU hits and resident rows cost what section 9's rows cost.

### Running it on a NUC

1. Build: `./tools/glm53/build-sycl.sh`. It also builds `$W/allocgate.so`.
2. Correctness: `python3 tools/glm53/serve_test.py`. It includes the disk runs: cold read, LRU hit and deadline NaN
   rows, both quants, store v1 and v2, bit-exact against `g53x check`.
3. Gate: `python3 tools/glm53/disk_test.py gate`. It builds `$W/tmp/disk_test/synthetic.k3xs` (22.5 GB) from
   `data/` unless `--store` is given. It ends with `GATE PASS: ...` or `GATE FAILED: <reasons>`.
4. Bench: `python3 tools/glm53/disk_test.py bench [--pieces 256,512,1024,4096] [--threads 4,8,16] [--iters 200]`. It
   prints the table above and writes `$W/tmp/disk_test/bench.json`.
5. Clean up: `python3 tools/glm53/disk_test.py clean` removes the synthetic store.

To use the real GLM store, mount it into the container: `DSYCL_ARGS="-v /path/to/store:/store:ro"`, then pass
`--store /store/glm.k3xs` (with its `.json` sidecar). With `--store`, rows are not compared against `g53x check`
(only the synthetic store's records have references), so the gate then checks allocations, faults and stats only. Serving for real:
`K3X_DISK_TIER=1 K3X_DISK_LRU_GB=8 g53x SET STORE - - serveidx PORT`.

## Files (`tools/glm53/`)

- `gguf_hdr.py URL OUT.json` (one shard's header via Range), `inventory.py HDRDIR` (classes, bytes per token, fit),
  `extract.py HDRDIR QUANT LAYERS OUTDIR` (experts 0-7 of the given layers).
- `glmbench.cpp` + `CMakeLists.txt`: ggml baseline and `--check` dumps. `glmref.cpp`: fp64 reference and comparisons.
- `g53x.cpp` + `g53x_ty.inc`: the iGPU kernels (`bench`, `check`, `serveidx`; `G53X_PROF=1` per-kernel times,
  `G53X_HOSTW=1` bench weights in host USM). `bw.cpp`: iGPU read roofline. `serve_test.py`: serveidx test (sections 9, 10).
  `disk_test.py gate|bench|clean` + `allocgate.c`: disk-tier allocation/fault gate and throughput bench (section 10).
- `g53c.cpp`: the CPU kernels. `bwcpu.c`: CPU read roofline.
- `build-cpu.sh`, `build-sycl.sh`, `dsycl.sh` (run in the oneAPI container with `/dev/dri`), `all.sh gpu|cpu`
  (tables above), `check.sh` (section 5), `asmstat.sh DUMPDIR REGEX` (instruction mix of IGC `.asm` dumps).
  Work dir `$W` (default `~/glm53-work`): `llama/` = the fork's tree, `data/` = `extract.py` output.
- `results/`: inventories, the two benchmark runs, the check output.

## 10. CUDA-exact CPU experts and g53cs (2026-10-29)

`tools/glm53/cpu/` adds AVX2/AVX-VNNI IQ2_XS, IQ3_XXS and IQ4_XS
kernels and a CPU-only `g53cs` server. The scalar oracle is the existing
`g53ref_exact.cpp`, whose GLM CUDA verification is recorded in
`tools/glm53/EXACT-RESULT.md`. The new kernels preserve its half q8 scales and
float butterfly sums, signed integer truncations, CUDA virtual-thread FMA and
reduction order, MUFU SwiGLU, FTZ and canonical NaN. Integer arithmetic and
independent virtual lanes are vectorized; there is no AVX-512 requirement.

The server accepts the k3x header and returns 6144-float expert rows. It uses
`expert_disk.h` for K3XS v1/v2 metadata and the preallocated disk tier, including
pinning until compute completes, deadlines and partial misses. Valid zero rows
are preserved. All request scratch and worker arenas are preallocated; startup
warms the workers and requires successful `mlockall`. Unlocked test mode must
be explicitly enabled with `G53_ALLOW_UNLOCKED=1`. Raw MUFU mappings add about
2.18 GB; there is no TANH dependency. See `tools/glm53/cpu/README.md` for the API
and environment variables. Submit is currently synchronous across two phases;
computation and row transmission do not overlap.

CPU-only offline validation:

```sh
tools/glm53/cpu/test.sh "$HOME/glm53-work/data" "$HOME/glm53-work/mufu"
python3 tools/glm53/cpu/test_server.py
G53_VNNI=' ' tools/glm53/cpu/test.sh
```

The first command checks 600 random/exceptional rows and all g/u/h/down words
for 3 extracted layers, 8 experts, 4 input cases and both worker counts. It also
counts allocations on caller and compute threads and reports warm page faults.
The second compares server wire bytes with independently generated scalar rows
at k=1/4/8/16, both store versions, heterogeneous gate/up types, missing and
unsupported IDs, duplicate disk IDs, partial misses and zero deadlines. It runs
the inherited disk-cache pin/LRU/read-failure/concurrency/allocation suite too.
Malloc-family and C++ new wrappers gate the real server request loop. Synthetic
and closed-form zero-gate server cases run without tables/model files; missing
external cases print SKIP explicitly. `.github/workflows/g53cs-test.yml` runs
these CPU-only tests without downloading model headers or data.

This sandbox forbids creating local sockets. Local integration therefore used
the same server request loop with a compile-time pipe transport; actual TCP is
selected automatically on unrestricted hosts/CI. Memory locking is unavailable
here, so the tests explicitly allowed unlocked operation. The final kernel run
reported zero allocations and zero warm minor/major faults for all six
layer/thread combinations. This does not establish a residency guarantee for
an unlocked process or certify the NUC's locked/TCP execution.

Measured on this laptop (Core Ultra 7 165H, Meteor Lake, shared host). Values
are **microseconds per expert**, including quantization, SwiGLU and worker
scheduling; they exclude sockets and disk I/O. Fast is the median of 7 runs,
scalar the median of 3, using the same number of workers and finite uniform
[-1,1] input. Experts 0..k-1 use the extracted weights; no replicated DRAM-sized
weight set is claimed. Workers were unpinned; these are indicative numbers.
Raw output: `tools/glm53/results/cpu-exact-mtl-165h.txt`.

| Layer | Threads | k | Fast us/expert | Scalar us/expert | Speedup |
|---|---:|---:|---:|---:|---:|
| L3 | 4 | 1 | 4849 | 28280 | 5.83x |
| L3 | 4 | 4 | 5076 | 23942 | 4.72x |
| L3 | 4 | 8 | 4698 | 27155 | 5.78x |
| L3 | 8 | 1 | 4080 | 25552 | 6.26x |
| L3 | 8 | 4 | 3817 | 20866 | 5.47x |
| L3 | 8 | 8 | 3585 | 24079 | 6.72x |
| L8 | 4 | 1 | 7774 | 31804 | 4.09x |
| L8 | 4 | 4 | 6459 | 32329 | 5.01x |
| L8 | 4 | 8 | 5730 | 31015 | 5.41x |
| L8 | 8 | 1 | 4458 | 20256 | 4.54x |
| L8 | 8 | 4 | 3793 | 21828 | 5.75x |
| L8 | 8 | 8 | 3748 | 19194 | 5.12x |
| L75 | 4 | 1 | 5644 | 27054 | 4.79x |
| L75 | 4 | 4 | 5455 | 24830 | 4.55x |
| L75 | 4 | 8 | 5794 | 29562 | 5.10x |
| L75 | 8 | 1 | 4226 | 20648 | 4.89x |
| L75 | 8 | 4 | 3868 | 23721 | 6.13x |
| L75 | 8 | 8 | 3385 | 18839 | 5.57x |

The improvement is 4.09–6.72x over the exact scalar pool. At 8 threads and k=8,
3.38–3.75 ms per expert still means 27–30 ms for eight experts. This does **not**
meet the earlier ~3 ms per-layer budget. Further row batching, IQ decoding and
activation/table prefetch work remain; there is no NUC speed claim.

NUC run instructions (Intel Core Ultra 5 225H, AVX2 + AVX-VNNI):

```sh
tools/glm53/cpu/build.sh
tools/glm53/cpu/test.sh "$HOME/glm53-work/data" "$HOME/glm53-work/mufu"
python3 tools/glm53/cpu/test_server.py
# Configure the service shell's memlock limit to cover arenas, cache and tables.
ulimit -l unlimited
export K3X_CPU_CPUS=0-7 K3X_MUFU="$HOME/glm53-work/mufu"
export K3X_DISK_TIER=1 K3X_DISK_LRU_GB=1 K3X_DISK_THREADS=4
export K3X_DISK_DEADLINE_MS=20 K3X_MAX_GB=24 K3X_FULLCHECK=1
unset G53_ALLOW_UNLOCKED
# IDX lines: gid gate_type down_type byte_offset; use the same full store here.
tools/glm53/cpu/g53cs /path/to/selected.idx /path/to/glm.k3xs 9000
```

Choose available worker CPUs from `lscpu -e`/the service affinity, reserve a CPU
for the request thread with `K3X_MAIN_CPU`, and repeat the benchmark at 4/8
workers on an otherwise idle NUC. The test driver accepts its 4/8-worker sets
internally; pin it with `taskset` if comparing core classes. Successful locked
startup, actual TCP rows and service latency remain NUC validation steps.
