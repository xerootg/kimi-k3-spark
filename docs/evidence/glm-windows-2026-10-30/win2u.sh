#!/bin/bash
# glm-legs.sh: GLM-5.3 cache-size / context legs on the Spark (native ~/llama-k3-s2), one server at a time.
# Each leg starts from the same frozen cache snapshot copy; warm request, then 2 timed 128-token requests.
cd /srv/glm-5.3; mkdir -p legs; cp snapshot/glm53-q2.k3sn legs/base.k3sn
P='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":128,"cache_prompt":false}'
leg() { name=$1 cache=$2 ctx=$3
  cp legs/${BASE:-base}.k3sn legs/$name.k3sn
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/$STORE LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/legs/$name.k3sn \
  LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 \
  LLAMA_EXPERT_STORE_RESERVE_MIB=4096 ${PROF:+GGML_CUDA_OP_PROFILE=$PROF} LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=$HL LLAMA_DSA_SHORT_CTX_SKIP=0 LLAMA_EXPERT_STORE_HOOK_LEAN=$HL LLAMA_EXPERT_STORE_HOOK_STATS=1 $BIN -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
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

STORE=glm53-q2-v1.k3xs
HL=0 leg CB0 80000 65536
HL=1 leg CB1 80000 65536
HL=0 leg CB0b 80000 65536
HL=1 leg CB1b 80000 65536
for l in HL0 HL1; do grep -a -h "host bank hooks:" legs/$l.log | tail -1 | cut -c1-200; done
echo WIN2U-DONE
