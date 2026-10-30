# Agent A notes (Spark side, spec, reference)

## 11:36 PDT — FIRST CRITICAL FINDING, read before writing kernels
- Production CUDA source tree = Spark ~/llama-k3-sc (all ggml-cuda sources byte-identical to upstream 547208d + patches at kimi-k3-spark cdba886). Kernel sources in the other trees are identical too (only host ggml-cuda.cu differs).
- **Kimi K3's routed-expert activation is NOT SwiGLU.** kimi-k3.cpp calls build_moe_ffn(..., LLM_FFN_SITU, ...):
    situ(gate, up) = beta*tanh(gate/beta) * sigmoid(gate) * lb*tanh(up/lb)      (lb part only if situ_linear_beta > 0)
  built as separate ggml ops: scale(1/beta) -> tanh -> scale(beta) ; sigmoid(gate) ; mul ; [scale(1/lb) -> tanh -> scale(lb)] ; mul.
  So the float ops to match are CUDA fast-math tanhf, sigmoid (1/(1+expf(-x))), scales, muls — plus whatever ggml-cuda fuses. tools/distributed/moebench.cpp uses ggml_swiglu_split, i.e. it does NOT reproduce production. I will build a corrected harness. Spec with exact instructions follows in ref/SPEC.md.
- Spark memory: MemAvailable is ~4.8 GB (valve at 4 GB), so my Spark runs will be tiny.

## 11:40 PDT — ref/SPEC.md v1 is up (read it, it supersedes the task text)
- Every float instruction pinned from SASS (ref/sass/*.sass). Summary: all FADD/FMUL/FFMA are .FTZ; accumulator
  init 0 and vec_dot fused as fma(dW, inner, acc); cross-warp ((w0+w1)+w2)+w3 then xor-butterfly 16,8,4,2,1.
- q8_1: d = amax * (float)(1/127) (FMUL, not div); q = trunc(RZ(v + copysign(0.5,v))) with v = MUFU_RCP(d)*x.
- SiTU chain: MUFU.TANH (x2), MUFU.EX2 (x1), MUFU.RCP (sigmoid + one per q8_1 block). No expf/division sequences.
- Next: MUFU tables (ref/mufu/), then dumps (ref/dumps/). ETA dumps ~12:20.

## 11:52 PDT — DUMPS READY: ref/rec_L1.bin, ref/rec_L11.bin, ref/dumps/{L1,L11}/, layout in ref/FORMAT.md
- 256 cases layer 1 (all IQ1_S), 160 cases layer 11 (IQ2_XXS gate/up, IQ1_S down), 16 experts per case,
  intermediates g, u, h and final dn rows from the real production kernels (nsys-verified identical kernel set).
- Next: MUFU tables (ref/mufu/) and the CPU reference (cpu/). Survey so far: MUFU.RCP depends only on the
  mantissa (one 2^23 table, |err| <= 1 ulp vs RN); MUFU.EX2 within [-2,+1] ulp; MUFU.TANH up to ~134 ulp and odd-
  symmetric, nontrivial only for |x| in [2^-12, 16): tanh(x) = x below, +-1 above (to be confirmed exhaustively).

## 12:03 PDT — MUFU TABLES READY, all three VERIFIED EXHAUSTIVELY on the GB10 (0 mismatches over all 2^32 inputs each)
- Layout + exact emulation code: ref/mufu/mufu_emu.h (plain C, __host__ __device__-able). Tables in ref/mufu/:
  rcp.bin  2^23 u32: MUFU.RCP(1.m); result = sign | (exp(t)+127-E)<<23 | mant(t); E==0 -> +-inf; inf -> +-0; NaN -> 0x7fffffff; e<1 -> +-0
  tanh.bin 15*2^23 u32: MUFU.TANH(+x), biased exponent E=115..129, index ((E-115)<<23)|mant; odd: sign | T. E<115 (incl +-0, denormals) -> x itself; E>129 and inf -> +-1.0; NaN -> 0x7fffffff
  ex2p.bin 31*2^23 u32: MUFU.EX2(+x) for E=103..133, index ((E-103)<<23)|mant; E<103 or E==0 -> 1.0; E>133 -> +inf
  ex2n.bin 33*2^23 u32: MUFU.EX2(-x) for E=102..134, index ((E-102)<<23)|mant; E<102 -> 1.0; E>134 (and -inf) -> +0
  md5: rcp 36f6a6fa6677286846396aecd641380a, tanh b08d3009bb3f76fb80fb2e94c470a2b2, ex2p 99045a62016120551506bddd9d57f231, ex2n c329e60e1a2e5606b632467f20633381
  (ex2n.bin may still be copying at 12:03 — check the md5 / size 1107296256 before use.)
- Compactness: RCP is compact (mantissa-only, 32 MiB raw, |delta vs RN| <= 1 ulp => 2 bits/entry possible).
  TANH and EX2 are NOT separable by exponent (no fixed-point 2^-22/23/24 table model fits EX2; TANH is non-monotone inside
  binades, steps -114..+211 ulp) so they are stored per binade over the nontrivial domain: 480 MiB + 2 GiB raw.

## 12:03 PDT — CPU REFERENCE IS BIT-EXACT ON EVERY STAGE (cpu/k3ref, source cpu/k3ref.cpp, static x86-64-v3 binary)
- Layer 1 (IQ1_S x3): g 12582912/12582912, u 12582912/12582912, h 12582912/12582912, dn 14680064/14680064 values; 4096/4096 rows.
- Layer 11 (IQ2_XXS gate/up, IQ1_S down): g 7864320/7864320, u same, h same, dn 9175040/9175040; 2560/2560 rows.
  All cases incl. edge cases (zeros, spikes, 1e-30, denormals, sigma 30/1000, zero blocks, constant).
- So the SPEC + ref/mufu tables are complete: B, plug mufu_emu.h (tanh/ex2 tables) into the SiTU stage and h/dn should match.
- Run: cd ~/k3-bitexact && ./cpu/k3ref validate ref/mufu ref/rec_L1.bin 32 19 19 19 ref/dumps/L1 8
       ./cpu/k3ref bench ref/mufu ref/rec_L1.bin 32 19 19 19 8 40
- Speed (nuc2, scalar emulation, not optimized): 8 threads k=1 8.7 ms, k=2 16.7, k=3 25.8, k=4 33.1 (median); IQ2_XXS layer k=1 9.7 ms.
  => only a correctness oracle as is, not a serving path (ggml's own CPU path was 0.48 ms at k=1).

## 12:11 PDT — production image check + fast CPU path + commits
- The production image llamacpp-spark:547208dca250-gcdba886 (sm_90 + sm_121a fatbin) has byte-identical normalized
  sm_121a SASS to the native build the dumps came from, for mmvq<IQ1_S>, mmvq<IQ2_XXS>, quantize_q8_1, softcap_f32,
  unary_gated_op_kernel<op_sigmoid>. So the dumps = production.
- cpu/k3ref with K3REF_FAST=1 (AVX2 int dots + MXCSR DAZ|FTZ native floats) is also bit-identical on all 416 cases;
  6.8 ms/expert at 8 threads (scalar 8.7). Oracle only.
- Committed + pushed kimi-k3-spark main: 55acd1c, 6acf5d1 (tools/bitexact/: SPEC.md, FORMAT.md, k3dump, mufu/, cpu/, sass/).
- 12:12 A -> B: bases confirmed: tanh.bin E0=115 (0x39800000), ex2p.bin E0=103 (0x33800000), ex2n.bin E0=102 (magnitude 0x33000000);
  exact rules in ref/mufu/mufu_emu.h (note tanh passes +-0/denormals through unchanged, ex2n beyond E=134 -> +0, ex2p beyond E=133 -> +inf).
  Making a third dump set now for the IQ1_S/IQ1_S/IQ2_XXS (down IQ2_XXS) layer kind -> ref/dumps/Lx (+ rec_Lx.bin).
- 12:14 ref/dumps/L83 + ref/rec_L83.bin (layer 83: gate/up IQ1_S, down IQ2_XXS; 96 cases, 16 experts). cpu/k3ref bit-identical on
  every stage (g/u/h 4718592/4718592, dn 5505024/5505024, rows 1536/1536), scalar and fast. All three layer kinds now covered.
