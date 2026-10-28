# Kimi K3 routed-expert FFN, 1-token decode on the Spark (GB10, sm_121a): exact float spec

Agent A, 2026-10-28. Source: upstream llama.cpp 547208d + kimi-k3-spark patches at cdba886 (byte-identical to
the Spark's native tree `~/llama-k3-sc`, which is the cdba886 A/B build). Build flags of ggml-cuda:
`-O3 -use_fast_math --generate-code=arch=compute_121a,code=[compute_121a,sm_121a]`, CUDA 13.0.
Every float instruction below was read from SASS (`cuobjdump -sass` of `~/llama-k3-sc/build/bin/libggml-cuda.so`;
extracts in `ref/sass/`). Kernel sequence confirmed from a production decode nsys trace (`~/k3prof.sqlite`).

## 0. Notation

All arithmetic is IEEE binary32 with round-to-nearest-even unless marked, and with **FTZ on every FADD/FMUL/FFMA**
(`.FTZ` on every instruction): denormal inputs are read as signed zero, a denormal result is written as signed
zero (sign preserved). `fma(a,b,c)` = single rounding. `h2f` = exact half->float. `f2h` = round-to-nearest-even
float->half (F2FP.F16.F32.PACK_AB, no FTZ; half subnormals are produced normally; overflow -> inf).
`MUFU_TANH`, `MUFU_EX2`, `MUFU_RCP` = the raw hardware special-function results (no range reduction around
them in these kernels); tables/emulation in `ref/mufu/`.

Model constants (GGUF): latent D = 3584 (gate/up K, down N), F = 3072 (gate/up N, down K),
situ_beta = 4.0, situ_linear_beta = 25.0. Layer types in the store: 80 layers IQ1_S/IQ1_S/IQ1_S
(e.g. layer 1), 10 layers IQ2_XXS/IQ2_XXS/IQ1_S (gate/up/down, e.g. layer 11), 2 layers IQ1_S/IQ1_S/IQ2_XXS.

## 1. The activation is SiTU, not SwiGLU

`kimi-k3.cpp` calls `build_moe_ffn(..., LLM_FFN_SITU, ...)`:

    situ(g, u) = beta*tanh(g/beta) * sigmoid(g) * lb*tanh(u/lb)     beta = 4, lb = 25

ggml graph ops: scale(g,1/4) tanh scale(4) | sigmoid(g) mul | scale(u,1/25) tanh scale(25) | mul.
ggml-cuda fuses them; the per-token kernel sequence of one routed layer (nsys) is:

| # | kernel | what |
|---|---|---|
| 1 | `quantize_q8_1` grid 14 | x (3584 f32) -> q8_1 |
| 2 | `mul_mat_vec_q<T_gate,1,false,true,false>` grid (768, n_exp) block (32,4) | gate rows, 4 rows per block |
| 3 | `softcap_f32` | a = tanh(g*0.25)*4 |
| 4 | `unary_gated_op_kernel<op_sigmoid,float>` | b = sigmoid(g) * a |
| 5 | `quantize_q8_1` grid 14 | x again (identical result) |
| 6 | `mul_mat_vec_q<T_up,1,false,true,false>` | up rows |
| 7 | `softcap_f32` | c = tanh(u*0.04)*25 |
| 8 | `k_bin_bcast<op_mul>` | h = b * c |
| 9 | `quantize_q8_1` grid (12, n_exp) | h (3072 per expert) -> q8_1 |
| 10 | `mul_mat_vec_q<T_down,1,false,true,false>` grid (896, n_exp) | down rows (unweighted output) |

Template args: `<type, ncols_dst=1, has_fusion=false, small_k=true, halve_iters=false>`. small_k is chosen because
blocks_per_row (14 or 12) < nwarps(4) * blocks_per_iter_1warp(4) = 16, so **rows_per_block = nwarps = 4**.
The split GEMV (LLAMA_EXPERT_STORE_SPLIT) only changes which slots are in the id list and gridDim.y; the per-row
arithmetic is the same kernel, so every expert row is a pure function of (x, that expert's weights).
The router weights are applied afterwards (`ggml_mul`) and are NOT part of the remote result.

## 2. quantize_q8_1 (block of 32 floats x[0..31] -> 32 int8 + half2 ds)

    amax = butterfly_max(|x_i|)            # FADD.FTZ |x|,-0 then FMNMX.FTZ; denormal |x| -> 0
    sum  = butterfly_add(x_i)              # see 2a
    d    = fmul(amax, 0x3C010204)          # amax * (float)(1/127) = 0.0078740157186985015869f, NOT a division
    if amax != 0:
        r = MUFU_RCP(d)
        v = fmul(r, x_i)                   # (FTZ)
        t = fadd_RZ(v, copysign(0.5f, v))  # FADD.FTZ.RZ: round toward zero
        q_i = (int8)(int32)trunc(t)        # FRND.TRUNC, F2I.TRUNC, low byte stored
    else q_i = 0
    ds = (f2h(d), f2h(sum))                # .x = d (low half), .y = sum (high half)

2a. butterfly_add over 32 lanes (lane i holds x_i): for off in 16,8,4,2,1: s_i = s_i + s_{i xor off}
(FADD.FTZ). Every lane ends with the same value; this equals the tree
`L1[j]=x[j]+x[j+16] (j<16); L2[j]=L1[j]+L1[j+8] (j<8); L3[j]=L2[j]+L2[j+4]; L4[j]=L3[j]+L3[j+2]; sum=L4[0]+L4[1]`.
butterfly_max is the same tree with fmaxf.

## 3. mul_mat_vec_q, one output row r of one expert (small_k, 4 warps x 32 lanes, 1 vec_dot per thread)

Thread t = 32*w + l (w = warp 0..3, l = lane 0..31) handles q-block kb = t/8 and sub-block j = t%8 (32 values
k = 256*kb + 32*j .. +31). Threads with kb >= blocks_per_row (14 for K=3584, 12 for K=3072) contribute 0.0f.

    p_t = fma(dW, inner, 0.0f)      # the accumulator starts at +0 and the vec_dot result is fused into it (FFMA)

  * IQ1_S (`vec_dot_iq1_s_q8_1`, iqs = j; y block = q8_1 block kb*8 + j):
      sumi  = sum over the 32 values of grid_int8 * q (exact int, IDP.4A); grid from iq1s_grid_gpu as in ggml
      dW    = fmul(h2f(blk.d), (float)(((qh[j] >> 11) & 0x0E) + 1))           # FMUL.FTZ
      delta = fma(-(float)(qh[j] & 0x8000) , 2^-17, -0.875)  = -0.875 or -1.125 exactly
      inner = fma(h2f(ds.x), (float)sumi, fmul(h2f(ds.y), delta))              # FMUL.FTZ then FFMA.FTZ
  * IQ2_XXS (`vec_dot_iq2_xxs_q8_1`, iqs = 2*j; y block = kb*8 + j):
      sumi  = exact int dot of the 32 signed grid values (iq2xxs_grid, ksigns via unpack_ksigns) with q
      sumi  = sumi * ls / 8   (C int division, truncates toward zero; ls = (aux32 >> 27) | 1)
      dW    = fmul(h2f(blk.d), h2f(ds.x))                                       # FMUL.FTZ
      inner = (float)sumi;   p_t = fma(dW, inner, 0.0f)

Cross-warp then intra-warp reduction (SASS order):

    s_l = ((p_{l} + p_{32+l}) + p_{64+l}) + p_{96+l}        # FADD.FTZ, warp0 + warp1 + warp2 + warp3
    row = butterfly_add over lanes l = 0..31 of s_l          # same tree as 2a (xor 16,8,4,2,1)

Note fma(a,b,+0) differs from a*b only in the sign of an exact zero (-0 -> +0) and in nothing else; the
+0.0f accumulator also turns -0 partials into +0.

## 4. Elementwise chain per expert (all FTZ)

    a  = fmul(MUFU_TANH(fmul(g, 0.25f)), 4.0f)               # softcap_f32: FMUL, MUFU.TANH, FMUL
    e  = MUFU_EX2(fmul(g, 0xBFB8AA3B))                      # -log2(e) = -1.44269502162933349609f
    s  = MUFU_RCP(fadd(e, 1.0f))                             # sigmoid
    b  = fmul(s, a)
    c  = fmul(MUFU_TANH(fmul(u, 0x3D23D70A)), 25.0f)        # 1.0f/25.0f = 0.0399999991059303283691f
    h  = fmul(b, c)                                          # k_bin_bcast op_mul

h (3072 per expert) is quantized with section 2 (96 blocks of 32), then the down matmul uses section 3 with
K = 3072 (12 q-blocks of 256) and N = 3584 rows. The output is the unweighted f32 row of 3584 values.

## 5. What is NOT exact-by-construction and needs tables

MUFU.TANH (2 per element of g/u), MUFU.EX2 (1), MUFU.RCP (1 in sigmoid, 1 per q8_1 block). All other ops are
IEEE RN + FTZ (and one RZ add) and are reproducible on any IEEE machine if done in exactly this order with
explicit fma where listed (compile with -ffp-contract=off, no -ffast-math, emulate FTZ).

The three MUFU functions are emulated by `mufu/mufu_emu.h` plus tables generated on the GB10 by
`mufu/mufu_tables gen` and verified by `mufu/mufu_tables verify` against the hardware for all 2^32 inputs of each
op: 0 mismatches for RCP, TANH and EX2 (2026-10-28).

- RCP depends only on the mantissa: one 2^23-entry table (32 MiB; within 1 ulp of RN, so 2 bits/entry would do),
  exponent applied exactly; zero/denormal -> +-inf, inf -> +-0, denormal results -> +-0, NaN -> 0x7fffffff.
- TANH is odd. |x| < 2^-12 (incl. zero and denormals, which are passed through): x. |x| >= 8: +-1. In between a
  per-binade table (15 binades, 480 MiB). It is not monotone inside a binade (neighbour steps -114..+211 ulp), so
  no simple interpolation model.
- EX2: x in [2^-24, 128) and (-256, -2^-25] per-binade tables (31 + 33 binades, 2 GiB); smaller |x| -> 1.0,
  x >= 128 -> +inf, x <= -256 -> +0. No fixed-point model 2^i * T[frac] with 22, 23 or 24 fraction bits and RN, RZ or
  RD conversion reproduces it (158M-865M mismatches of 1.79G tested), so it is not compact either. Error vs
  correctly rounded exp2 is within [-2, +1] ulp everywhere.

Verified end to end: `cpu/k3ref` (scalar C++, this spec + tables) reproduces every g, u, h and dn value of the
reference dumps (416 cases, layer 1 and layer 11) bit for bit.
