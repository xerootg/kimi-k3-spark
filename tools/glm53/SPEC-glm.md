# GLM-5.3 routed decode on GB10 (sm_121a, CUDA -use_fast_math)

Status: source-derived candidate, **not yet certified bit-exact with GLM CUDA dumps**.
The q8_1 lowering was already read from this build's SASS for K3 (tools/bitexact/SPEC.md §2).
The IQ integer operations and launch/reduction rules below come from the production sources in ref/.
The IQ multiply/accumulate contraction and SwiGLU lowering still require the SASS checks in §7.
Do not authorize model offloading on the strength of the local zero tests alone.

## 1. Graph and scope

D=6144, F=2048, eight selected experts per token. GLM calls build_moe_ffn with LLM_FFN_SILU
(ref/glm-dsa.cpp:458-466). Separate up and gate projections use mul_mat_id
(ref/llama-graph.cpp:2371-2391), followed by ggml_swiglu_split(gate,up)
(ref/llama-graph.cpp:2400-2425), then down mul_mat_id (ref/llama-graph.cpp:2483).
No SiTU, softcap, router weights, shared expert, final expert sum, or residual belongs in this remote result.
This spec assumes the ordinary unscaled, unbiased, unclamped expert tensors; the graph's optional
clamp branch is ref/llama-graph.cpp:2403-2419, and must not be enabled for this graph.

The inventory's 75 ordinary routed layers contain (gate,up,down):
71 x (IQ2_XS,IQ2_XS,IQ3_XXS), one x (IQ3_XXS,IQ3_XXS,IQ4_XS), three x (IQ2_XS,IQ2_XS,IQ4_XS).
Their ggml ids are 17,18,23; block sizes 74,98,136 bytes, QK=256
(ref/ggml-common.h:387-412, ref/ggml-common.h:454-460; lookup tables start at :627, :1017, :1120).
The inventory ALSO lists MTP layer 78 (Q2_K/Q2_K/Q3_K), outside these 75 layers.
It remains unsupported by the exact CPU/SYCL implementation, and serveidx rejects its ids with NaN
so the engine computes them locally. See §6 for its source arithmetic; do not call this MTP support.
Q4_K/Q5_K/Q6_K in UD-Q4_K_XL, attention and shared experts are outside this implementation.

## 2. Arithmetic and q8_1

add/mul/fma below mean binary32 RN-even, FTZ inputs/results with signed zero preserved;
fma rounds once. Half conversion is RN-even, preserves half subnormals and overflows to infinity.
Integer divisions truncate toward zero, including negative dividends. Do not replace integer scale
arithmetic with dequantized float weights. Build host/SYCL code with -ffp-contract=off -fno-fast-math,
using explicit fma at the specified accumulation sites. All addresses are little-endian ggml blocks.

For each consecutive block of 32 input floats, quantize_q8_1 does:

    a_i = abs(x_i); s_i = x_i
    for off = 16,8,4,2,1, simultaneously:
        a_i = max_FTZ(a_i, a_(i xor off))
        s_i = add(s_i, s_(i xor off))
    d = mul(a_0, float_bits(0x3c010204))   # RN(1/127)
    r = MUFU_RCP(d)
    q_i = a_0 == 0 ? 0 : low_byte(F2I_TRUNC(add_RZ(mul(r,x_i),copysign(.5,mul(r,x_i)))))
    ds = (half(d), half(s_0))

Source: ref/ggml-cuda/quantize.cu:54-103; shuffle trees:
ref/ggml-cuda/common.cuh:469-475, :529-538. The division by constant becomes multiplication;
xi/d becomes RCP then multiplication, roundf uses FADD.FTZ.RZ then truncation (existing K3 SASS).
For finite v in the quantizer range this is round-to-nearest, ties away from zero.
The shared implementation computes that rounding from v's bits, without fp64 or device round intrinsics.
An exact implementation stores BOTH scale and float sum as half, then reads them back as float.
Neither a float scale without half rounding nor d*sum(q) is CUDA's ds.
There are 192 x blocks, and 64 h blocks per expert; no padding values for these shapes.
The ordinary IQ dots use ds.x only; restoring ds.y is nevertheless required for the q8_1 contract.

## 3. IQ vec dots (one CUDA thread invocation = 32 weights)

In the following j=0..7 within a 256-weight block; y is q8 block 8*kb+j.
Decode table entries as signed int8 lanes. Each DP4A adds four exact signed byte products to int32.
Seven sign bits encode eight signs: bit7 = parity(bits0..6). Negate the selected table bytes.
The source's unpack_ksigns accepts eight bits and corrects bit7 with parity, which is equivalent
(ref/ggml-cuda/vecdotq.cuh:97-104).

IQ2_XS: iqs=2*j. Four uint16 at qs[4*j..4*j+3] contain grid index (low 9 bits,
iq2xs_grid, eight bytes) and seven sign bits. Dot the first two signed grids against y.q[0..15]
to produce integer A, and the last two against y.q[16..31] to produce B.
Let l0=scales[j]&15, l1=scales[j]>>4. Then:

    I = (A*l0 + B*l1 + (A+B)/2) / 4       # TWO truncating integer divisions
    d = mul(h2f(weight.d), h2f(y.ds.x))
    return mul(d, float(I))               # may contract with caller's accumulator

Every decode/index/integer step and return: ref/ggml-cuda/vecdotq.cuh:1143-1183.
The old g53x float formula .25*((l0+.5)*A+(l1+.5)*B) is not equivalent.

IQ3_XXS: iqs=2*j. Eight byte indices at qs[8*j..8*j+7] select four-byte iq3xxs_grid
entries. aux32 is the little-endian word qs[64+4*j..67+4*j]; four groups of seven sign bits
sign each pair of grids. Dot all 32 signed bytes with y.q -> S (int32), then:

    l = aux32 >> 28
    I = (l*S + S/2) / 2                    # truncating integer divisions, before float conversion
    d = mul(h2f(weight.d), h2f(y.ds.x))
    return mul(d, float(I))

Source: ref/ggml-cuda/vecdotq.cuh:1232-1273. A float (l+.5)*.5 scale changes the result.

IQ4_XS: iqs=4*j. Sixteen bytes qs[16*j..16*j+15] select low-nibble table values for
y.q[0..15], high-nibble values for y.q[16..31] (signed kvalues_iq4nl table).
Accumulate all 32 products into S, then:

    l = ((scales_l[j/2] >> (4*(j&1))) & 15) | (((scales_h >> (2*j)) & 3) << 4)
    I = S * (l-32)                        # integer multiply, NOT a float scale
    d = mul(h2f(weight.d), h2f(y.ds.x))
    return mul(d, float(I))

Source: ref/ggml-cuda/vecdotq.cuh:1417-1447.

## 4. MMVQ layout and reduction

For all three IQ types qi/vdr = 8: IQ2_XS and IQ3_XXS qi=16,vdr=2;
IQ4_XS qi=32,vdr=4 (ref/ggml-common.h:145-167 and vecdotq.cuh:1143,:1232,:1417).
GB10 uses four warps for one column (ref/ggml-cuda/mmvq.cu:452-465,:557-582).
ID gathers prohibit halve_iters (ref/ggml-cuda/mmvq.cu:1415-1422).
The small_k condition is nb < 16; IQ3_XXS explicitly disables it even on GB10
(ref/ggml-cuda/mmvq.cu:1377-1411). Thus:

| stage/type | nb=K/256 | small_k | rows/block | block | grid for 8 ids |
|---|---:|---|---:|---|---|
| gate/up IQ2_XS | 24 | false | 1 | (32,4) | (2048,8,1) |
| gate/up IQ3_XXS | 24 | false | 1 | (32,4) | (2048,8,1) |
| down IQ3_XXS | 8 | false (exception) | 1 | (32,4) | (6144,8,1) |
| down IQ4_XS | 8 | true | 4 | (32,4) | (1536,8,1) |

Rows/block: ref/ggml-cuda/mmvq.cu:595-615; grid/block: :1270-1280.
Small_k still has four warp contributions per row; it changes rows/block, not the reduction.
For each output row, virtual t=32*w+lane (0..127) starts at kb=t/8, j=t%8,
then visits kb+=16 while kb<nb. Therefore gate/up's first 64 threads accumulate two dots,
last 64 threads one dot; down's first 64 threads one dot, remaining threads zero.
Start each thread at +0 and execute, in increasing kb:

    p_t = fma(mul(h2f(weight[kb].d),h2f(y[8*kb+j].d)), float(I), p_t)

Source loop: ref/ggml-cuda/mmvq.cu:634-648,:754-756,:780-794.
This contraction is a SASS hypothesis for GLM IQ types, consistent with K3's verified kernels.
Do not fuse the first scale multiply into another operation.
Then combine warps in order and apply the 32-lane XOR tree:

    s_lane = add(add(add(p_lane,p_(32+lane)),p_(64+lane)),p_(96+lane))
    for off=16,8,4,2,1: s_lane=add(s_lane,s_(lane xor off))  # simultaneous

Sources: ref/ggml-cuda/mmvq.cu:797-851; ref/ggml-cuda/common.cuh:469-475.
The scalar implementation computes the lane-zero tree (same operand order).
No SYCL reduce_over_group is allowed in this path. A 16-lane Intel subgroup does not reproduce
CUDA simply by striding 32-value blocks and reducing them in a subgroup.

## 5. SwiGLU and fusion

Source SiLU is g/(1+expf(-g)) (ref/ggml-cuda/unary.cuh:98-100).
Candidate CUDA -use_fast_math lowering, with RN+FTZ at the arithmetic sites:

    z = mul(g, float_bits(0xbfb8aa3b))     # -log2(e)
    e = MUFU_EX2(z)
    r = MUFU_RCP(add(1,e))
    h = mul(mul(g,r),u)

No TANH needed. RCP, EX2 positive and EX2 negative are ALREADY tabulated and exhaustively
verified on GB10 by tools/bitexact/mufu/mufu_tables.cu; use mufu_emu.h, rcp.bin (32 MiB),
ex2p.bin (992 MiB), ex2n.bin (1056 MiB). The existing 480 MiB TANH table is not used.
Do not substitute expf, std::exp2 or sycl::native::exp, nor IEEE correctly rounded division.

Two mul_mat_id + split GLU can fuse to a single has_fusion=true MMVQ:
ref/ggml-cuda/ggml-cuda.cu:4893-4934, eligibility :2476-2504. For this graph types match,
input is f32, decode batch=1, weights have WEIGHTS buffer usage; the conditions are satisfied.
The fused kernel computes gate and up with identical dot/reduction arithmetic; it adds
+0 bias placeholders even when no bias exists, then applies SiLU(gate)*up
(ref/ggml-cuda/mmvq.cu:853-882). That add may normalize -0. The candidate treats both
inputs as add(value,+0) for h. Dumped g/u themselves are unfused MMVQ outputs.
Unfused GLU also applies SiLU then multiply (ref/ggml-cuda/unary.cu:306-321,:401-403).
No fused down+SwiGLU and no q8_1+MMVQ fusion is present in this CUDA path:
quantization is launched separately (ref/ggml-cuda/mmvq.cu:1898).
The SYCL implementation may recompute quantization inside a row; it must produce the same bits.
Actual fusion selection must still be confirmed by the owner's production decode trace.

## 6. Patches and MTP boundary

patches/0002-zero-copy-host-experts.patch changes CUDA weight access/dispatch.
patches/0003-expert-store.patch touches mmvq.cu: 64-bit sample/channel weight addressing
(ref/ggml-cuda/mmvq.cu:742-751), adds Hopper Q6_K TMA code, and expert-store/split scheduling.
It does not change these IQ vec-dot functions, quantize.cu or unary math.
The Q6_K TMA path is not this GB10 IQ path. Use the patched sources, not generic upstream
launch guesses; the ref/ files used above already contain the relevant patches.

MTP Q2_K/Q3_K (ids 10/11): both vdr=2, qi=16, eight thread slices per 256 block;
ref/ggml-cuda/vecdotq.cuh:363-396,:449-481,:894-970 defines the exact integer indexing.
For each invocation, four i=0..3 shifts each dot eight q8 values (two DP4As).
Q2: v_i=(v>>(2*i))&0x03030303 per word; extract scale low nibble and minimum high nibble;
A_i=dot(v_i,u_i), B_i=sum(u_i) via DP4A(0x01010101,u_i).
Accumulate sumD += d8_i*(A_i*scale_i), sumM += d8_i*(B_i*min_i),
return weight.d*sumD - weight.dmin*sumM, in source order. This uses local integer sums,
not q8 ds.y. Q3: combine two low bits and the high-mask bit, subtract 4 per value;
extract signed 6-bit scale (packed low4/high2 minus32), multiply the integer dot by scale,
accumulate sum += d8_i*(dot_i*scale_i), return weight.d*sum.
These float accumulations and subtract contractions need their own SASS; the IQ candidate
must never be silently used for them. The exact implementation fails closed for those types.

## 7. Required Spark certification

Run glmdump against ~/llama-k3-sc/build/bin/libggml-cuda.so, preserving production flags,
then compare every g/u/h/dn word with g53ref_exact; not just final dn or an error tolerance.
The dump format is K3 FORMAT.md with D=6144,F=2048,K=8, ids are record slots, case-major LE.
Separate graphs end at g/u/h/dn so intermediate output flags cannot suppress production fusion.
Cases 0..7 are zeros, spikes, tiny normals, signed denormals, sigma30, zero blocks, constant.25,
sigma1000; the rest are seeded normals at six scales with eight outliers. Non-finite inputs
and NaN payload propagation still need targeted certification in addition to these finite cases.

Capture nsys kernel names/grid/block for all three ordinary layer combinations; verify fused
<type,1,true,false,false> gate/up, IQ3 down <18,1,false,false,false>, IQ4 down small_k=true.
Disassemble the actual production sm_121a cubin with cuobjdump -sass. Check precisely:
1. IQ2/3/4 integer scale divides are signed truncation (including negative odd intermediates).
2. Half-to-float scale multiply is FMUL.FTZ; final I multiply is fused into tmp as FFMA.FTZ,
   including BOTH gate/up loop iterations, with no reassociation of d or partial sums.
3. Cross-warp FADD order is w0,w1,w2,w3; SHFL.BFLY offsets 16,8,4,2,1.
4. Fused SiLU uses FMUL.FTZ with 0xbfb8aa3b, MUFU.EX2, FADD.FTZ +1,
   MUFU.RCP, FMUL.FTZ gate*rcp then FMUL.FTZ *up; check exceptional exp/div guards,
   FTZ, +0 bias adds, and whether any reciprocal correction or different ordering appears.
5. q8_1 uses the same known K3 SASS (FMUL constant, MUFU.RCP, FADD.FTZ.RZ,
   half packing without FTZ), for both x and h tensor layouts.
Any discrepancy requires updating this spec and the shared CPU/SYCL implementation, then
rerunning all stage comparisons. Current code is a candidate, not evidence those checks passed.
