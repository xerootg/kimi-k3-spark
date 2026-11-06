#!/bin/bash
# Vision window: production args + --mmproj (BF16). /props modalities, text md5 gate, then vision_client.py. glm53-gh200 at 0.
set -u
cd /srv/glm-5.3-flash; out=win; BIN=$HOME/llama-el/build/bin/llama-server
pgrep -f "[l]lama-server -m " >/dev/null && { echo ABORT; exit 1; }
BASE=$(sed -n "s/^BASE=\"\(.*\)\"$/\1/p" flashwin2.sh)
cp snapshot/glm53flash-q4-full.k3sn $out/snap-v.k3sn
(while sleep 2; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; done) > $out/mem-v.txt & mon=$!
env $BASE LLAMA_EXPERT_STORE_HBM_MIB=40960 LLAMA_EXPERT_STORE_VALVE_FLOOR=10 LLAMA_EXPERT_STORE_MAX_BATCH=4096 LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3-flash/$out/snap-v.k3sn $BIN -m q4/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --mmproj mmproj/mmproj-BF16.gguf --alias glm-5.3-flash --host 127.0.0.1 --port 18104 --ctx-size 1048576 --parallel 1 --batch-size 4096 --ubatch-size 4096 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 189360 --cache-ram 98304 --temp 1.0 --top-p 0.95 --threads 16 --threads-batch 32 --flash-attn on --jinja --metrics --no-warmup --fit off --load-mode mmap -lv 4 > $out/server-v.log 2>&1 & pid=$!
for i in $(seq 1 200); do curl -s -m 2 127.0.0.1:18104/health | grep -q ok && break; kill -0 $pid || { echo SERVER-DIED; grep -a -i -E "error|fail" $out/server-v.log | tail -3; exit 1; }; sleep 3; done
echo READY; grep -a -E "clip|mmproj|vision|image_max_pixels|image_min_pixels|projector" $out/server-v.log | grep -a -v "^\s*$" | head -12 | cut -c1-180
curl -s 127.0.0.1:18104/props | python3 -c "import json,sys;d=json.load(sys.stdin);print('props modalities',d.get('modalities'))"
curl -s 127.0.0.1:18104/completion -H "Content-Type: application/json" -d '{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":128,"cache_prompt":false}' | python3 -c "import json,sys,hashlib;d=json.load(sys.stdin);print('text gate md5',hashlib.md5(d['content'].encode()).hexdigest()[:10],round(d['timings']['predicted_per_second'],1),'tok/s')"
echo "GPU used after load+text: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
python3 -I vision_client.py 18104 | tee $out/vision.jsonl
echo "PEAK GPU $(awk "{print \$2}" $out/mem-v.txt | sort -n | tail -1) MiB"
kill -INT $pid; for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -KILL $pid 2>/dev/null; kill $mon; rm -f $out/snap-v.k3sn
pgrep -f "[l]lama-server -m " >/dev/null && echo WARN-running || echo no-server
echo VWIN-DONE
