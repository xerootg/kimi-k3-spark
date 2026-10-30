# Routing trace collector

These files mirror the new-file hunks in `patches/0003-expert-store.patch`.
That patch installs the collector into the engine's `tools/k3-pfdump/` and
registers its CMake target `llama-k3-pfdump`. The Spark Dockerfile copies and
applies patches; it does **not** copy this directory. It builds the tool but
does not package it in the runtime image. Keep the patch mirror synchronized
(the offline test checks this).

For an already patched native engine, from this checkout on Spark:

```bash
cp tools/k3-pfdump/{CMakeLists.txt,k3-pfdump.cpp,trace.h} "$HOME/llama-k3-bs/tools/k3-pfdump/"
cmake --build "$HOME/llama-k3-bs/build" --target llama-k3-pfdump -j 8
"$HOME/llama-k3-bs/build/bin/llama-k3-pfdump" --check decode.bin
```

`--check FILE` needs no model or GPU initialization. It streams one step at a
time, prints per-layer real/predictor/pair counts and top-k intersection / k,
then aggregate precision. Scores and IDs may arrive in either order. Pairing
is by `(step, destination layer)`; the first MoE layer normally has no predictor.
Missing layers show up as count differences; absent predictions have precision
`n/a`. Bad framing, duplicate records/IDs and nonfinite scores fail the check.
Counts cannot prove model completeness without knowing its expected MoE layers.

Collection reads architecture-prefixed GGUF block/expert/top-k/dense counts and
subtracts `nextn_predict_layers` from `block_count`. Dense and MTP callbacks are
excluded; prefill is disabled by `step=-1`. The shared `build_moe_ffn` real
selection is `ffn_moe_topk-<layer>`, not `ffn_moe_slots`, the split position maps,
`moe_prefetch_self`, or `moe_prefetch` (which is a predicted slot lookup).
`build_moe_cache_prefetch` names scores `moe_prefetch_probs-<destination>`.
Kind 3 contains int32 IDs; kind 1 contains float32 scores. Kinds 2/4/5/6..9
retain their historical meanings and bytes. `--ignore-eos` forces the requested
number of single-token decode iterations, including any generated EOS IDs.

Offline regression (C++ compiler and Python standard library only):

```bash
python3 -m unittest discover -s tools/k3-pfdump -p 'test_*.py' -v
```
