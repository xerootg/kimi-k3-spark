# Bit-exact remote expert compute: Spark-side spec, MUFU tables, reference dumps, CPU reference (2026-10-28)

Can a box other than the Spark compute Kimi K3's routed-expert FFN for one decode token and return rows that are
bit-identical to the Spark's CUDA kernels? This directory holds the Spark side of that study: the exact
float-level spec of the production kernels, emulation tables for the GPU's special-function unit, reference
dumps from the real kernels, and a scalar CPU reference that reproduces them bit for bit. The SYCL kernel for
the NUC's Arc iGPU is in `sycl/`.

- `SPEC.md`: the routed-expert path of a 1-token decode on the GB10 (sm_121a, -use_fast_math), every float
  instruction pinned from SASS. Kimi K3's routed experts use SiTU (beta 4, linear beta 25), not SwiGLU, so
  `tools/distributed/moebench.cpp` (swiglu) does not compute what production computes.
- `sass/`: `cuobjdump -sass` extracts of the production kernels (libggml-cuda.so of the cdba886 native build).
- `mufu/`: `mufu_emu.h` (C emulation of MUFU.RCP / MUFU.TANH / MUFU.EX2 from tables), `mufu_tables.cu`
  (`gen DIR` writes the tables on a GB10 in ~1 s, `verify DIR` checks the emulation against the hardware for all
  2^32 inputs of each op), `mufu_probe*.cu` (the structure surveys). The tables are 2.6 GB and not in git:
  Spark `~/k3bx/mufu/`, nuc2 `~/k3-bitexact/ref/mufu/` (md5 rcp 36f6a6fa..., tanh b08d3009..., ex2p 99045a62...,
  ex2n c329e60e...).
- `k3dump.cpp` + `extract.py`: pull a few store records with small preads and dump x, ids, gate rows, up rows,
  SiTU output and down rows from the production kernels (link against the production tree's libggml and load
  its libggml-cuda.so; nsys shows the same kernels and fusions as llama-server). Layout in `FORMAT.md`.
  `g++ -O2 -std=c++17 -I <tree>/ggml/include k3dump.cpp -o k3dump -L <tree>/build/bin -lggml -lggml-base`
- `cpu/k3ref.cpp`: scalar reference. `g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math -pthread
  -march=x86-64-v3 -static -I <llama.cpp>/ggml/src -I mufu k3ref.cpp -o k3ref`;
  `k3ref validate MUFU_DIR RECORDS N TG TU TD DUMP_DIR [threads]`, `k3ref bench MUFU_DIR RECORDS N TG TU TD threads iters`.
- `cpu/k3cpu.{h,cpp}`: the serving-speed CPU version (AVX2 + AVX-VNNI library, thread pool, P-core gather and E-core
  scalar-load kernels), bit-identical on all dumps; ~0.25 ms per expert on nuc3's 8 E-cores. See `cpu/README.md`.

Results: MUFU emulation 0 mismatches over 2^32 inputs for each of the three ops; k3ref bit-identical on every
value of every stage of 512 dumped cases (256 of layer 1, all IQ1_S; 160 of layer 11, IQ2_XXS gate/up and IQ1_S
down; 96 of layer 83, IQ1_S gate/up and IQ2_XXS down; 16 experts each, edge cases included), in both its scalar oracle path and its `K3REF_FAST=1` path (AVX2
integer dots, native float ops under MXCSR DAZ|FTZ, hardware fmaf). Speed on nuc2, 8 threads, one token: scalar
8.7 / 16.7 / 25.8 / 33.1 ms for k = 1..4 experts, fast 6.8 / 13.5 / 19.3 / 26.5 ms, so it is an oracle and not a
serving path (ggml's own CPU kernels: 0.48 ms at k=1).

The production image (`llamacpp-spark:547208dca250-gcdba886`, multi-arch sm_90 + sm_121a) has the same sm_121a
SASS as the native build used for the dumps for mul_mat_vec_q<IQ1_S|IQ2_XXS,1,0,1,0>, quantize_q8_1, softcap_f32
and unary_gated_op_kernel<op_sigmoid> (normalized diff: 0 lines).
