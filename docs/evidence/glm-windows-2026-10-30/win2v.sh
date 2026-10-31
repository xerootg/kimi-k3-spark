#!/bin/bash
# win2v.sh (Spark): (1) the single-file GB10 GEMV bench; (2) a clean tree ~/llama-k3-c = pinned llama.cpp + the committed
# stack (0001-0006, 0009, 0012) + 0011 v2 (DSA short-ctx skip, fixed) + 0012b (hook stats split) + 0012c (prefetch wide),
# all candidates off by default; (3) DSA v2 A/B with the adopted knobs on (SZ=1, HL=1): DS 0/1/0/1 standard legs and the
# 8192-token identity check.
cd ~/win2 && export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -std=c++17 -arch=sm_121a -lineinfo gb10-gemv-bench.cu -o gb10-gemv-bench > gemv-build.log 2>&1; echo "gemv-build=$?"; grep -a -E "error|warning: .*fast" gemv-build.log | head -3 | cut -c1-200
if [ -x ./gb10-gemv-bench ]; then ./gb10-gemv-bench --list 2>&1 | tail -8 | cut -c1-200; timeout 900 ./gb10-gemv-bench - 128 3 > gb10-gemv-run.log 2>&1; echo "gemv-run=$?"; grep -a -i -E "MISMATCH|MATCH|GB/s|ms/token|Q5_K|Q8_0|Q6_K|Q4_K|class|SM" gb10-gemv-run.log | head -80 | cut -c1-200; fi
nvcc -O3 -std=c++17 -arch=sm_121a -lineinfo gb10-bankread-bench.cu -o gb10-bankread-bench > bankread-build.log 2>&1; echo "bankread-build=$?"; grep -a -E "error" bankread-build.log | head -3 | cut -c1-200
if [ -x ./gb10-bankread-bench ]; then timeout 600 ./gb10-bankread-bench > gb10-bankread-run.log 2>&1; echo "bankread-run=$?"; grep -a -i -E "MISMATCH|MATCH|GB/s|ms|variant|[A-F]:|SM|L2|canMap|pageable|unified" gb10-bankread-run.log | head -60 | cut -c1-200; fi
SHA=547208dca250ceb3e4c8615598d043f2b602a527
rm -rf ~/llama-k3-c; git clone -q ~/llama-k3-bs ~/llama-k3-c && cd ~/llama-k3-c && git checkout -q $SHA && [ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "ABORT-WINDOW clone/checkout"; exit 1; }
for p in ~/win2/stack/000[1-6]-*.patch ~/win2/stack/0009-*.patch ~/win2/stack/0012-*.patch ~/win2/0011-dsa-short-ctx.v2.patch ~/win2/0012b-hook-stats-split.patch ~/win2/0012c-prefetch-wide.patch; do git apply "$p" || { echo "ABORT-WINDOW apply failed $p"; exit 1; }; done; echo applied-ok
export CUDACXX=/usr/local/cuda/bin/nvcc
nice -n 19 cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121 -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF > ~/llama-k3-c-cmake.log 2>&1 && nice -n 19 cmake --build build -j 12 --target llama-server > ~/llama-k3-c-build.log 2>&1; echo "buildc=$? $(date +%H:%M)"
[ -x build/bin/llama-server ] || { echo "ABORT-WINDOW build"; grep -a -E "error" ~/llama-k3-c-build.log | head -5 | cut -c1-200; exit 1; }
BIN=~/llama-k3-c/build/bin/llama-server
cd /srv/glm-5.3; mkdir -p legs; cp snapshot/glm53-q2.k3sn legs/base.k3sn
P='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":128,"cache_prompt":false}'
leg() { name=$1 cache=$2 ctx=$3
  cp legs/base.k3sn legs/$name.k3sn
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/legs/$name.k3sn \
  LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 \
  LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 \
  LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_HOOK_STATS=1 LLAMA_DSA_SHORT_CTX_SKIP=$DS \
  $BIN -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
  --host 127.0.0.1 --port 18080 --ctx-size $ctx --parallel 1 --batch-size 2048 --ubatch-size 512 -ngl 999 --n-cpu-moe 999 \
  --moe-cache-mib $cache --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 5 \
  > legs/$name.log 2>&1 & pid=$!
  until curl -s -m 2 127.0.0.1:18080/health | grep -q ok; do sleep 3; kill -0 $pid 2>/dev/null || { echo "$name: server died: $(grep -a -E 'GGML_ASSERT|error' legs/$name.log | head -1 | cut -c1-160)"; return; }; done
  for i in 0 1 2; do curl -s -m 900 127.0.0.1:18080/completion -H 'Content-Type: application/json' -d "$P" > legs/$name.r$i.json; done
  for i in 1 2; do python3 -c "import json,hashlib,sys;d=json.load(open(sys.argv[1]));t=d['timings'];print(sys.argv[2],round(t['predicted_per_second'],3),'tok/s md5',hashlib.md5(d['content'].encode()).hexdigest()[:10])" legs/$name.r$i.json "$name r$i" 2>&1 | tail -1; done
  grep -a -E "dsa.*skip|short.ctx|host bank hooks" legs/$name.log | tail -3 | cut -c1-200
  kill -INT $pid; wait $pid 2>/dev/null; sleep 5
}
DS=0 leg DS0 80000 65536
DS=1 leg DS1 80000 65536
DS=0 leg DS0b 80000 65536
DS=1 leg DS1b 80000 65536
for ds in 0 1; do cp legs/base.k3sn legs/L$ds.k3sn
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/legs/L$ds.k3sn LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_DSA_SHORT_CTX_SKIP=$ds $BIN -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size 2048 -ngl 999 --n-cpu-moe 999 --moe-cache-mib 80000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 > legs/L$ds.log 2>&1 & pid=$!
  until curl -s -m 2 127.0.0.1:18080/health | grep -q ok; do sleep 3; kill -0 $pid 2>/dev/null || { echo "L$ds died"; break; }; done
  (cd ~/win2 && timeout 2400 python3 tests/measure-prefill.py --url http://127.0.0.1:18080 --tokens 8192 --server-log /srv/glm-5.3/legs/L$ds.log --output ~/win2/dsa2-long-$ds.json > ~/win2/dsa2-long-$ds.out 2>&1); python3 -c "import json,hashlib;d=json.load(open('$HOME/win2/dsa2-long-$ds.json'));print('L$ds 8192-token prompt: decode',round(d['timings']['predicted_per_second'],3),'tok/s md5',hashlib.md5(d['content'].encode()).hexdigest()[:10],round(d['timings']['prompt_per_second'],1),'pp tok/s')" 2>&1 | tail -1
  kill -INT $pid; wait $pid 2>/dev/null; sleep 3; done
echo WIN2V-DONE
