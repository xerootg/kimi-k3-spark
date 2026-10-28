# k3x: bit-exact routed-expert FFN on the Intel Arc iGPU (SYCL)

Standalone SYCL implementation of one Kimi K3 decode token's routed-expert FFN (q8_1 of x, gate/up MMVQ,
SiTU, q8_1 of h, down MMVQ) that reproduces the Spark's production CUDA kernels bit for bit, following
`../SPEC.md` (FTZ on every FADD/FMUL/FFMA, explicit FMAs, CUDA's 4-warp reduction order, MUFU.RCP/EX2/TANH from
the exhaustive tables in `../mufu/`). Not built on ggml-sycl.

- `k3x_math.h`: shared host/device arithmetic (q8_1, IQ1_S / IQ2_XXS vec_dot tails, row reduction, FTZ helpers).
- `k3x_mufu.h`: MUFU emulation from the tables (same layout and edge rules as `../mufu/mufu_emu.h`).
- `k3x_act.h`: the SiTU chain (SPEC section 4).
- `k3x.cpp`: device kernels (SLM-staged weight tiles, dp4a integer dots, sub-group shuffle reduction in the CUDA
  order), host reference, and the modes below.
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

Types: IQ1_S = 19, IQ2_XXS = 16. Results 2026-10-28 are in `docs/experiments.md`.
