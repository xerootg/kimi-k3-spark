# f53s: GLM-5.3-Flash expert server for the NUCs (window nd)

`f53s.cpp` computes GLM-5.3-Flash routed expert rows on a NUC's CPU with the GB10's arithmetic, bit for bit, for
the batched remote expert tier (`LLAMA_EXPERT_STORE_REMOTE_COLS=1`, see docs/experiments.md "NUC distribution,
batched across streams"). It covers Flash's 39 regular expert layers (gate/up IQ2_XS 4096x2048, down IQ3_XXS
2048x4096, SwiGLU clamped at 10); layers 11, 12 and 44 (IQ4_XS down: the GB10 takes its small-K path) and the MTP
layer are not served.

Build (static, AVX2 + AVX-VNNI, from the checkout root):

```sh
g++ -O3 -std=c++17 -pthread -static -march=x86-64-v3 -mavxvnni -ffp-contract=off -fno-fast-math \
    -Itools/bitexact tools/glm53/flash/f53s.cpp -o f53s
```

Check against the GPU: `flashrow.cpp` (docs/evidence/glm-windows-2026-10-30/nuc-batched/) runs the production ggml
CUDA kernels on 8 records and random inputs and writes the rows; `f53s check recs8.bin flashrow.bin` compares every
row bit for bit (`F53_MUFU` names the verified GB10 rcp/ex2p/ex2n tables). `ndclient.py` replays the same cases
over the wire against a running server.

Serve: `f53s serve <records.bin> <records.idx> <port>` with the box's records (gate|up|down, 8,060,928 bytes each)
and an index of `gid offset` lines (gid = layer * 288 + expert); `mkown.py` writes both from the Spark's store, and
the engine's `LLAMA_EXPERT_STORE_REMOTE_SET` file. Env: `F53_MUFU` (required), `F53_CPUS` (the caller takes the
first, workers the rest), `F53_SPIN_US` (idle spin before futex sleep). The tables are read into locked anonymous
memory (a file mapping is page cache, which loading 30 GB of records evicts, and the first requests then take major
faults); the records are read with the page cache dropped behind and locked as far as RLIMIT_MEMLOCK allows. The
request path allocates nothing; the log reports page faults since start and compute time per pair.

Row kernel: `fast_row2` (default) reduces each block's eight VNNI dot products with one transposed `hadd` tree
instead of sixteen horizontal sums (integer adds: the same bits); `F53_ROW=1` selects the original `fast_row`.
On nuc2 (12 workers): one pair 0.30 ms (was 0.42), eight pairs 2.6 ms (was 3.1).

State (window nuc5, 2026-11-10): on the production configuration (five batch-invariant slots) the tier is byte-exact
with patch 0067 (the engine sends magic `0x4E450000` in batch-invariant mode and the server then sums each row in the
one-warp MoE order) and adds +2.0-3.3 % at four clients and +3.3-3.5 % at five for about 50-60 W of NUC package power. It
stays off in production; see docs/experiments.md "NUC expert tier on five batch-invariant slots". The evidence
directory `docs/evidence/glm-windows-2026-10-30/nuc5/` has the server start script (`nucserve5.sh`) and the legs.
