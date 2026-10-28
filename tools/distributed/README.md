# Distributed-inference study tools (2026-10-26)

Used for the "Spark + the Intel NUCs over the Thunderbolt mesh" entry in `docs/experiments.md`.

- `netbench.py`: TCP bulk throughput (`bulk HOST PORT SECS STREAMS`) and request/response latency
  (`rr HOST PORT REQ RESP COUNT`) against `serve PORT`. Python only, no dependencies.
- `remotesim.py`: replays the real top-16 expert ids from `k3-pfdump` dumps through the Spark's LRU plus a
  static remote expert set (`--train=a.bin[,b.bin] --eval=c.bin --sizes=... --skips=...`). Pure Python.
- `moebench.cpp` + `CMakeLists.txt`: one layer's routed-expert FFN for one token on any ggml backend, from
  real expert-store records; `serve:PORT` answers k ids + a 3584-float activation with k rows.
  CPU: `cmake -S . -B build -DGGML_SRC=<llama.cpp>/ggml -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON
  -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON` in ubuntu:24.04. SYCL: same with `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON
  -DGGML_SYCL_DNN=OFF -DGGML_CPU=OFF -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx` inside
  `intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04`; `bundle.sh` (run in that image with the tree at
  /w) collects the SYCL/Level Zero runtime and the compute-runtime user-mode driver next to the binary. Run with
  `GGML_BACKEND_PATH=lib/libggml-sycl.so LD_LIBRARY_PATH=lib` and the render group.
- `rrclient.c`: the Spark-side client (`gcc -O2`), fixed k or k drawn from a histogram (`dist p0,p1,...`);
  `RR_IDX=<server index>` sends real ids (one layer per request) and counts NaN replies.

Remote experts in the engine (patch 0003, `LLAMA_EXPERT_STORE_REMOTE`, 2026-10-28):

- `remoteset.py`: picks the static remote set from the k3-pfdump dumps (most selected (layer, expert) pairs after
  skipping the hottest, which the LRU keeps anyway, up to a byte budget from the store's record sizes) and replays
  a held-out dump through the LRU for several skips. `--out` writes "layer expert" lines. `--boxes
  name:GB:speed,...` places the pool over several boxes by layer (fan-out 1 except at a box's capacity edge, the
  densest layers on the fastest box), `--configs "a;b;c"` compares configurations (NVMe misses, remote rows,
  fan-out, per-box request sizes), `--out-prefix` writes one set file per box. pfdump-code2 holds the same tokens
  as pfdump-code: evaluate out of domain with `--train=pfdump-prose.bin --eval=pfdump-code.bin`.
- `remotecompose.py SRC_IDX SRC_BIN SET OUT_IDX [--subset]`: a box's file from an existing export (no store
  reads), or `--subset` an index into SRC_BIN for a server on the same box.
- `remoteexport.py STORE SET IDX [MB/s] > records.bin`: the set's records (gate | up | down, store order, no
  padding) with O_DIRECT reads, throttled, plus the index `<global_id> <gate_type> <down_type> <byte_offset>`,
  global_id = layer * n_expert + expert. Piped through `nc` to the remote box (46 GB in ~5 min at 160 MB/s).
- `remote-standin.c`: a stand-in server for plumbing and timing (busy-waits base + k * per-k us, replies zero rows;
  output differs from the gold one by construction).
