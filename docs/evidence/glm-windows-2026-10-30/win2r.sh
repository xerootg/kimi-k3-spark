#!/bin/bash
# glm-legs.sh: GLM-5.3 cache-size / context legs on the Spark (native ~/llama-k3-s2), one server at a time.
# Each leg starts from the same frozen cache snapshot copy; warm request, then 2 timed 128-token requests.
cd /srv/glm-5.3; mkdir -p legs; cp snapshot/glm53-q2.k3sn legs/base.k3sn
P='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":128,"cache_prompt":false}'
leg() { name=$1 cache=$2 ctx=$3
  cp legs/${BASE:-base}.k3sn legs/$name.k3sn
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/$STORE LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/legs/$name.k3sn \
  LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 \
  LLAMA_EXPERT_STORE_RESERVE_MIB=4096 ${PROF:+GGML_CUDA_OP_PROFILE=$PROF} LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=0 ${POL:+LLAMA_EXPERT_STORE_POLICY=$POL} ${POL:+LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144} $BIN -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
  --host 127.0.0.1 --port 18080 --ctx-size $ctx --parallel 1 --batch-size 2048 --ubatch-size 512 -ngl 999 --n-cpu-moe 999 \
  --moe-cache-mib $cache --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 \
  > legs/$name.log 2>&1 & pid=$!
  until curl -s -m 2 127.0.0.1:18080/health | grep -q ok; do sleep 3; kill -0 $pid 2>/dev/null || { echo "$name: server died"; return; }; done
  for i in 0 1 2; do curl -s -m 900 127.0.0.1:18080/completion -H 'Content-Type: application/json' -d "$P" > legs/$name.r$i.json; done
  for i in 1 2; do python3 -c "import json,hashlib,sys;d=json.load(open(sys.argv[1]));t=d['timings'];print(sys.argv[2],round(t['predicted_per_second'],3),'tok/s md5',hashlib.md5(d['content'].encode()).hexdigest()[:10])" legs/$name.r$i.json "$name r$i"; done
  grep -a -E "predictor top-k|predicted misses|wait at the real lookup|hit rate" legs/$name.log | tail -4 | cut -c1-220
  echo "$name MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
  kill -INT $pid; wait $pid 2>/dev/null; sleep 5
}





BIN=~/llama-k3-bs/build/bin/llama-server

cd ~/llama-k3-bs && git apply --check ~/win2/0018-lfu-policy.patch && git apply ~/win2/0018-lfu-policy.patch && export CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH && nice -n 19 cmake --build build -j 12 --target llama-server > ~/win2/build18.log 2>&1; echo "build18=$?"; [ "$(tail -c 200 ~/win2/build18.log | grep -c "Built target\|Linking")" -ge 1 ] || { echo "ABORT-WINDOW: patch or build failed"; exit 1; }; cd /srv/glm-5.3
STORE=glm53-q2-v1.k3xs
POL= leg POL0 80000 65536
POL=lfu leg POL1 80000 65536
POL= leg POL0b 80000 65536
POL=lfu leg POL1b 80000 65536
for l in POL0 POL1; do grep -a -h -E "hit rate|wait at the real lookup" legs/$l.log | tail -2 | sed -E "s/^[0-9.]+ I [a-z_]+: expert store: //" | cut -c1-120; done
echo WIN2R-DONE
