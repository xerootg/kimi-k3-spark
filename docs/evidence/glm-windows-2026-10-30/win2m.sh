#!/bin/bash
# win2m.sh (Spark): nsys per-kernel trace of steady GLM decode on the clean tree ~/llama-k3-c (SZ=1, HL=1), short context
# (standard prompt, 64 tokens) and long context (8192-token prompt then 64 tokens). The first attempt (window n) used
# --capture-range=cudaProfilerApi, which llama-server never triggers, so no report was written. This one uses
# nsys launch + start/stop around the timed request only. Output: nsys/glm-{short,long}.nsys-rep and stats CSVs.
BIN=~/llama-k3-c/build/bin/llama-server; [ -x $BIN ] || BIN=~/llama-k3-bs/build/bin/llama-server
cd /srv/glm-5.3; export PATH=/usr/local/cuda/bin:$PATH; mkdir -p nsys; echo "bin=$BIN"
P64='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":64,"cache_prompt":false}'
for mode in short long; do cp legs/base.k3sn legs/NS$mode.k3sn; nsys sessions list 2>/dev/null | grep -q glm$mode && nsys stop --session=glm$mode >/dev/null 2>&1
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/legs/NS$mode.k3sn LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 \
    nsys launch --session-new=glm$mode --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
    $BIN -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size 2048 -ngl 999 --n-cpu-moe 999 --moe-cache-mib 80000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 > legs/NS$mode.log 2>&1 & pid=$!
  until curl -s -m 2 127.0.0.1:18080/health | grep -q ok; do sleep 3; kill -0 $pid 2>/dev/null || { echo "NS$mode died: $(grep -a -i -E 'GGML_ASSERT|error' legs/NS$mode.log | head -1 | cut -c1-160)"; break; }; done
  if [ $mode = short ]; then curl -s -m 900 127.0.0.1:18080/completion -H 'Content-Type: application/json' -d "$P64" > /dev/null
    nsys start --session=glm$mode -o /srv/glm-5.3/nsys/glm-$mode --force-overwrite=true > legs/NS$mode.nsys.out 2>&1; sleep 2
    curl -s -m 900 127.0.0.1:18080/completion -H 'Content-Type: application/json' -d "$P64" > legs/NS$mode.r1.json
    nsys stop --session=glm$mode >> legs/NS$mode.nsys.out 2>&1
    python3 -c "import json,hashlib;d=json.load(open('legs/NS$mode.r1.json'));print('short',round(d['timings']['predicted_per_second'],2),'tok/s md5',hashlib.md5(d['content'].encode()).hexdigest()[:10])" 2>&1 | tail -1
  else (cd ~/win2 && timeout 2400 python3 tests/measure-prefill.py --url http://127.0.0.1:18080 --tokens 8192 --server-log /srv/glm-5.3/legs/NSlong.log --output ~/win2/ns-long.json > ~/win2/ns-long.out 2>&1)
    # the long prompt is now in the slot's KV; trace a 64-token continuation (cache_prompt true keeps the 8192 tokens)
    PL=$(python3 -c "import json;d=json.load(open('$HOME/win2/ns-long.json'));print(json.dumps({'prompt':d.get('prompt_text') or d.get('prompt') or '','temperature':0,'seed':1,'n_predict':64,'cache_prompt':True}))" 2>/dev/null)
    nsys start --session=glm$mode -o /srv/glm-5.3/nsys/glm-$mode --force-overwrite=true > legs/NS$mode.nsys.out 2>&1; sleep 2
    [ -n "$PL" ] && curl -s -m 2400 127.0.0.1:18080/completion -H 'Content-Type: application/json' -d "$PL" > legs/NS$mode.r1.json
    nsys stop --session=glm$mode >> legs/NS$mode.nsys.out 2>&1
    python3 -c "import json;d=json.load(open('legs/NS$mode.r1.json'));t=d['timings'];print('long',round(t['predicted_per_second'],2),'tok/s after',t.get('prompt_n'),'prompt tokens (cached',d.get('tokens_cached'),')')" 2>&1 | tail -1
  fi
  kill -INT $pid; wait $pid 2>/dev/null; sleep 5
  ls -la nsys/glm-$mode.nsys-rep 2>/dev/null | awk '{print $5, $9}'; tail -2 legs/NS$mode.nsys.out | cut -c1-160
  nice -n 19 nsys stats --report cuda_gpu_kern_sum --format csv -o nsys/glm-$mode nsys/glm-$mode.nsys-rep > /dev/null 2>&1; nice -n 19 nsys stats --report cuda_api_sum --format csv -o nsys/glm-$mode nsys/glm-$mode.nsys-rep > /dev/null 2>&1; nice -n 19 nsys stats --report cuda_gpu_trace --format csv -o nsys/glm-$mode nsys/glm-$mode.nsys-rep > /dev/null 2>&1; nice -n 19 nsys stats --report osrt_sum --format csv -o nsys/glm-$mode nsys/glm-$mode.nsys-rep > /dev/null 2>&1
  ls -la nsys/ | grep $mode | awk '{print $5, $9}'
done
echo WIN2M-DONE
