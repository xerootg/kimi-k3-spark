# k3x: bit-exact routed-expert FFN on the Intel Arc iGPU (SYCL)

Standalone SYCL implementation of one Kimi K3 decode token's routed-expert FFN (q8_1 of x, gate/up MMVQ,
SiTU, q8_1 of h, down MMVQ) that reproduces the Spark's production CUDA kernels bit for bit, following
`../SPEC.md` (FTZ on every FADD/FMUL/FFMA, explicit FMAs, CUDA's 4-warp reduction order, MUFU.RCP/EX2/TANH from
the exhaustive tables in `../mufu/`). Not built on ggml-sycl.

- `k3x_math.h`: shared host/device arithmetic (q8_1, IQ1_S / IQ2_XXS vec_dot tails, row reduction, FTZ helpers).
- `k3x_mufu.h`: MUFU emulation from the tables (same layout and edge rules as `../mufu/mufu_emu.h`).
- `k3x_act.h`: the SiTU chain (SPEC section 4).
- `k3x.cpp`: device kernels, host reference, and the modes below. Current matvec ("v7"): records are repacked at
  load so 4 lanes share each output row (lane = kb0 group, CUDA's first two xor-tree levels as lane shuffles),
  IGC's native DP4A (`__builtin_IB_dp4a_ss`; `dot_acc` compiles to 32-bit multiply emulation on this iGPU), grid
  tables and activation in SLM, 512-item work-groups. Older variants stay behind macros (MV2..MV4, V5, V6).
- `build.sh`: build recipe (oneAPI 2026.1.1 container, `-ffp-model=precise -ffp-contract=off -fno-fast-math`).

Run on nuc2 from a directory holding `k3x` and the runtime `lib/` (see `build.sh`), with the tables in
`K3X_MUFU` (default `../ref/mufu`), as `sudo -u xero -g render env LD_LIBRARY_PATH=lib ./k3x ...`:

```
k3x RECORDS N TG TD validate DUMPDIR [ncases]   # vs agent A's dumps (x/ids/g/u/h/dn), counts per stage
k3x RECORDS N TG TD check [n]                   # device vs host reference on random inputs ("synthetic" records ok)
k3x RECORDS N TG TD bench k1,k2,.. ITERS        # p50/p90 per token (K3X_PROF=1: per-kernel times)
k3x RECORDS N TG TD serve PORT                  # tools/distributed/rrclient.c protocol, ids = record slots
k3x IDX BIN - - serveidx PORT                   # remote-set server: IDX "<gid> <tg> <td> <offset>", BIN records
```

Serving: `serve`/`serveidx` busy-poll the socket and stream each expert's row as soon as its down matmul is done
(`K3X_NOSTREAM=1`: one write). `serveidx` caps the GPU-resident records at `K3X_MAX_GB` (default 24), writes the served
(layer, expert) list to `K3X_SETOUT`, self-checks one token at startup and sends NaN for any all-zero row: above
roughly 34-42 GB resident on the Arc 130T, kernel submissions sometimes fail with no error and leave rows zero.

Types: IQ1_S = 19, IQ2_XXS = 16. Results 2026-10-28 are in `docs/experiments.md`.
