Changes:
- tools/k3-pfdump/k3-pfdump.cpp:32: actual eval callback validates model-sized
  single-token tensors; kind 3 = ffn_moe_topk-* int32, kind 1 =
  moe_prefetch_probs-* float32 with the unchanged destination-layer suffix.
- tools/k3-pfdump/k3-pfdump.cpp:107: architecture-prefixed GGUF block_count,
  expert_count, expert_used_count, leading_dense_block_count, nextn_predict_layers;
  no K3 dimensions in collection code. GLM becomes layers 3..77, 256, top-8;
  dense 0..2 and MTP 78 excluded. --ignore-eos forces 2048 decode iterations.
- tools/k3-pfdump/trace.h:43: --check FILE runs before model/backend init;
  per-layer real/predictor/pair counts and intersection/k precision, keyed by
  step+destination, independent of callback ordering; malformed traces fail.
  L3 normally has no predictor: n/a, excluded from precision denominator.
- patches/0003-expert-store.patch:7683: mirrored engine source/CMake/header.
- tools/k3-pfdump/test_callback.cpp:19 and test_pfdump.py:13: compile the actual
  callback offline, compare K3/GLM bytes with legacy writer, reject dense/MTP/
  prefill/slot lookups, test checker and place.py default parser interoperability.
- tools/glm53/PLACEMENT.md:23 and tools/k3-pfdump/README.md:1: build/check docs.

Source provenance and Spark build (commands to run there, not executed here):
Dockerfile docker/llamacpp-spark.Dockerfile:27 copies patches; :30 applies them.
Patch 0003 creates tools/k3-pfdump and registers target llama-k3-pfdump.
It is NOT copied from this tools directory by Docker. Runtime copy list (:49)
omits the dumper. On an existing 0001–0006 + GLM-predictor native engine:
```bash
cp tools/k3-pfdump/{CMakeLists.txt,k3-pfdump.cpp,trace.h} "$HOME/llama-k3-bs/tools/k3-pfdump/"
cmake --build "$HOME/llama-k3-bs/build" --target llama-k3-pfdump -j 8
```
CMake names the binary llama-k3-pfdump; a Spark k3-pfdump is a local rename/alias.

2048-token collection command, from ~/win2 on Spark with the engine stopped:
```bash
cd "$HOME/win2"
env -u LLAMA_EXPERT_STORE_REMOTE -u LLAMA_EXPERT_STORE_REMOTE_SET \
 -u LLAMA_EXPERT_STORE_TRACE -u LLAMA_EXPERT_STORE_PREFETCH_DUMP \
 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs \
 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 \
 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_GPU_SLOTS=1 \
 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_PREFETCH=1 \
 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=8 \
 LLAMA_EXPERT_STORE_PREFETCH_MAX=16 \
 "$HOME/llama-k3-bs/build/bin/llama-k3-pfdump" \
 -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
 -ngl 99 --n-cpu-moe 99 --moe-cache-mib 80000 -c 65536 -t 8 -tb 20 \
 -b 512 -ub 512 -ctk f16 -ctv f16 \
 -f "$HOME/win2/glm-prompts.txt" -n 2048 --ignore-eos --out "$HOME/win2/glm-decode.bin"
"$HOME/llama-k3-bs/build/bin/llama-k3-pfdump" --check "$HOME/win2/glm-decode.bin"
```
Uncertainty: production YAML absent. Above reconstructs settings from
 evidence/glm-decode-pf3.log (v1 store, prefetch max 16), README topology and
 documented 80000/64k; exact current production CLI/model path cannot be certified.
No full engine/GPU build or real trace run here; no network/commit/push.
Tests: python3 -m unittest discover -s tools/k3-pfdump -p 'test_*.py' -v
Last lines: Ran 2 tests; steps=2 layers=75 pairs=148 precision=100.000% (1184/1184); OK
K3 test: PASS 896 experts: callback stream equals legacy bytes; prefill/dense/MTP excluded
Tests: python3 -m unittest discover -s tools/glm53 -p 'test_place.py' -v
Last lines: Ran 4 tests in 0.582s; OK. git diff --check: clean.
