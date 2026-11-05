#!/bin/bash
# Production args (cache-ram 98304, U=4096, tier 48128, elastic KV, spare slots). 1) prompt-cache restore: A, B, A again.
# 2) a ~900k-token prompt (elastic KV grows to ~17 GB in HBM, the valve trims the tier). glm53-gh200 at 0.
set -u
cd /srv/glm-5.3-flash; out=win; BIN=$HOME/llama-el/build/bin/llama-server
pgrep -f "[l]lama-server -m " >/dev/null && { echo ABORT; exit 1; }
BASE=$(sed -n "s/^BASE=\"\(.*\)\"$/\1/p" flashwin2.sh)
cp snapshot/glm53flash-q4-full.k3sn $out/snap-long3.k3sn
(while sleep 2; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) $(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)"; done) > $out/mem-long3.txt & mon=$!
env $BASE LLAMA_EXPERT_STORE_HBM_MIB=40960 LLAMA_EXPERT_STORE_VALVE_FLOOR=10 LLAMA_EXPERT_STORE_MAX_BATCH=4096 LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3-flash/$out/snap-long3.k3sn $BIN -m q4/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --alias glm-5.3-flash --host 127.0.0.1 --port 18104 --ctx-size 1048576 --parallel 1 --batch-size 4096 --ubatch-size 4096 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 189360 --cache-ram 98304 --temp 1.0 --top-p 0.95 --threads 16 --threads-batch 32 --flash-attn on --jinja --metrics --no-warmup --fit off --load-mode mmap -lv 4 > $out/server-long3.log 2>&1 & pid=$!
for i in $(seq 1 200); do curl -s -m 2 127.0.0.1:18104/health | grep -q ok && break; kill -0 $pid || { echo SERVER-DIED; exit 1; }; sleep 3; done; echo READY
python3 -I - <<"PY"
import hashlib,json,random,time,urllib.request
W=("kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap helm chart argocd sync health probe container image registry runner pipeline postgres backup barman wal archive snapshot etcd raft").split()
def text(n,seed): r=random.Random(seed); return " ".join(r.choice(W) for _ in range(int(n/1.29)))
def go(label,p,n=32,cache=True):
    t0=time.time(); d=json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18104/completion",data=json.dumps({"prompt":p,"temperature":0,"n_predict":n,"cache_prompt":cache,"ignore_eos":True}).encode(),headers={"Content-Type":"application/json"}),timeout=7200).read()); t=d["timings"]
    print(json.dumps({"req":label,"prompt_n":t["prompt_n"],"cache_n":t.get("cache_n"),"prefill_tps":round(t["prompt_per_second"],1),"prefill_s":round(t["prompt_ms"]/1e3,1),"decode_tps":round(t["predicted_per_second"],2),"wall_s":round(time.time()-t0,1),"md5":hashlib.md5(d["content"].encode()).hexdigest()[:10]}),flush=True)
A="Summarize log A:\n"+text(24000,7); B="Summarize log B:\n"+text(24000,8)
go("A",A); go("B",B); go("A-again",A); go("A-step",A+"\nTool output: "+text(300,11))
go("1M-cold","Summarize this huge log:\n"+text(1035000,321),n=64,cache=False)
go("after-1M-std","Write a Python quicksort function and explain it.",n=128,cache=False)
PY
grep -a -E "memory valve: .*released|elastic KV \(ubatch\)" $out/server-long3.log | tail -3 | cut -c1-220
grep -a -E -i "restor|prompt cache|cache_ram|state" $out/server-long3.log | grep -a -v "snapshot" | tail -4 | cut -c1-200
kill -INT $pid; for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -KILL $pid 2>/dev/null; kill $mon
rm -f $out/snap-long3.k3sn
echo "PEAK GPU $(awk "{print \$2}" $out/mem-long3.txt | sort -n | tail -1) MiB, LOWEST host MemAvailable $(awk "{print \$3}" $out/mem-long3.txt | sort -n | head -1) MiB"
pgrep -f "[l]lama-server -m " >/dev/null && echo WARN-running || echo no-server
echo LONGWIN3-DONE
