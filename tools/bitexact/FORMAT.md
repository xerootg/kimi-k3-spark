# Reference dumps (agent A, 2026-10-28)

Produced on the Spark by `k3dump.cpp` (this dir) linked against the production tree `~/llama-k3-sc`
(libggml-cuda.so md5 9b6875fb3ca6a917def821df65ab9dd4 = upstream 547208d + kimi-k3-spark patches at cdba886,
sm_121a, -use_fast_math). nsys confirmed the kernels are exactly the production decode ones (SPEC.md section 1):
quantize_q8_1 grid 14, mul_mat_vec_q<T,1,0,1,0> grid (768,16) block (32,4), softcap_f32,
unary_gated_op_kernel<op_sigmoid>, k_bin_bcast<op_mul>, quantize_q8_1 grid (12,16), mmvq down grid (896,16).
Every output was also recomputed once per 50 cases: bit-identical (deterministic).

## Records
- `rec_L1.bin`: 32 records of layer 1 (gate IQ1_S, up IQ1_S, down IQ1_S), 6451200 bytes each =
  gate 2150400 | up 2150400 | down 2150400. Store expert ids 3 + 28*i (i = 0..31). Slot i = i-th record.
- `rec_L11.bin`: 16 records of layer 11 (gate IQ2_XXS, up IQ2_XXS, down IQ1_S), 7827456 bytes each =
  gate 2838528 | up 2838528 | down 2150400. Store expert ids 5 + 55*i (i = 0..15).
- Inside a record each matrix is the GGUF row layout: gate/up = 3072 rows x 3584 cols (14 blocks of 256 per row),
  down = 3584 rows x 3072 cols (12 blocks per row).
- `rec_L83.bin`: 16 records of layer 83 (gate IQ1_S, up IQ1_S, down IQ2_XXS), 7139328 bytes each =
  gate 2150400 | up 2150400 | down 2838528. Store expert ids 7 + 55*i (i = 0..15). Block structs = ggml's block_iq1_s (50 B) / block_iq2_xxs (66 B).

## Dumps: dumps/L1 (256 cases), dumps/L11 (160 cases), dumps/L83 (96 cases)
All little-endian, raw arrays, case-major:
- `x.bin`   f32 [ncase][3584]          the routed-expert input (latent activation)
- `ids.bin` i32 [ncase][16]            slots into the record file (L1: 16 of 32, random order; L11: a permutation of 16)
- `g.bin`   f32 [ncase][16][3072]      gate rows (mmvq output, before any activation), row j = expert ids[j]
- `u.bin`   f32 [ncase][16][3072]      up rows
- `h.bin`   f32 [ncase][16][3072]      SiTU output fed to the down matmul: h = (4*tanh(g/4)*sigmoid(g)) * (25*tanh(u/25))
- `dn.bin`  f32 [ncase][16][3584]      final unweighted down rows (what a remote must return, bit for bit)
- `kind.txt` case -> description. Cases 0-7 are edge cases (zeros, two spikes, 1e-30 normals, denormal inputs,
  sigma 30, zero blocks, constant 0.25, sigma 1000); the rest are N(0, s^2) with s in {0.02,0.1,0.3,1,3,10} plus
  8 heavy-tail outlier channels.
The q8_1 activations are not dumped separately (no hook in the backend); they are fully determined by x / h and
SPEC.md section 2, and the cpu/ reference reproduces g/u (which depend on them) bit for bit.
