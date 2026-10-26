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
- `rrclient.c`: the Spark-side client (`gcc -O2`), fixed k or k drawn from a histogram (`dist p0,p1,...`).
