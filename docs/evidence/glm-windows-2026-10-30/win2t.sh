#!/bin/bash
# win2t.sh (Spark): patch 0005b (bounded wait on active speculative reads; no knob): 3 x 128-token legs on the current
# tree with the patch applied; compare with window r's POL0 legs (same tree minus 0005b). md5 gate 9449ae5738.
cd ~/llama-k3-bs && git apply --check ~/win2/0005b-bounded-spec-wait.patch && git apply ~/win2/0005b-bounded-spec-wait.patch && export CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH && nice -n 19 cmake --build build -j 12 --target llama-server > ~/win2/build5b.log 2>&1; echo "build5b=$?"; [ "$(tail -c 200 ~/win2/build5b.log | grep -c 'Built target\|Linking')" -ge 1 ] || { echo "ABORT-WINDOW: patch or build failed"; exit 1; }
cd /srv/glm-5.3; P='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":128,"cache_prompt":false}'
for n in 1 2; do tag=SB$n; cp legs/base.k3sn legs/$tag.k3sn
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/legs/$tag.k3sn LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 ~/llama-k3-bs/build/bin/llama-server -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size 2048 -ngl 999 --n-cpu-moe 999 --moe-cache-mib 80000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 > legs/$tag.log 2>&1 & pid=$!
  until curl -s -m 2 127.0.0.1:18080/health | grep -q ok; do sleep 3; kill -0 $pid 2>/dev/null || { echo "$tag died"; break; }; done
  for i in 0 1 2; do curl -s -m 900 127.0.0.1:18080/completion -H 'Content-Type: application/json' -d "$P" > legs/$tag.r$i.json; done
  for i in 1 2; do python3 -c "import json,hashlib,sys;d=json.load(open(sys.argv[1]));print(sys.argv[2],round(d['timings']['predicted_per_second'],3),'tok/s md5',hashlib.md5(d['content'].encode()).hexdigest()[:10])" legs/$tag.r$i.json "$tag r$i"; done
  kill -INT $pid; wait $pid 2>/dev/null; sleep 3; done
echo WIN2T-DONE
