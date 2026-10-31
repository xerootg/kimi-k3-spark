#!/bin/bash
# win2q.sh (Spark): prompt cache for the RE-workbench pattern. An 8192-token prompt A, then a different 2048-token prompt B,
# then A again: time to first token of the second A with --cache-ram 0 (today) vs 8192 MiB (the parked slot restored).
# Production build ~/llama-k3-bs; ubatch 2048; same snapshot per leg.
cd ~/win2; W=~/win2; M=/srv/glm-5.3/models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf; E=/srv/glm-5.3/store/glm53-q2-v1.k3xs
for cr in 0 8192 0 8192; do tag=CR$cr$RANDOM; cp $W/base.k3sn $W/$tag.k3sn
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$E LLAMA_EXPERT_STORE_SNAPSHOT=$W/$tag.k3sn LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 \
    ~/llama-k3-bs/build/bin/llama-server -m $M --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size 2048 -ngl 999 --n-cpu-moe 999 --moe-cache-mib 80000 --cache-ram $cr --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 > $W/$tag.log 2>&1 & PID=$!
  until curl -s -m 2 127.0.0.1:18080/health | grep -q ok; do sleep 3; kill -0 $PID 2>/dev/null || { echo "$tag died"; break; }; done
  python3 - "$cr" <<'PY'
import json,sys,time,urllib.request,hashlib,random
cr=sys.argv[1]
def gen(n,seed):
    r=random.Random(seed); lines=[]; k=0
    while sum(len(l)+1 for l in lines)<n*4: lines.append(f"uint64_t f{k}(uint64_t x) {{ return ((x ^ 0x{r.getrandbits(32):08x}ULL) + {k}) >> {k%13}; }}"); k+=1
    return "\n".join(lines)
def req(p,n=32):
    t=time.time(); body=json.dumps({"prompt":p,"n_predict":n,"temperature":0,"seed":1,"cache_prompt":True}).encode()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18080/completion",body,{"Content-Type":"application/json"}),timeout=3000); d=json.loads(r.read()); tm=d["timings"]
    return time.time()-t, tm.get("prompt_n"), tm.get("cache_n"), round(tm.get("prompt_ms",0)/1000,1), hashlib.md5(d["content"].encode()).hexdigest()[:8]
A=gen(8192,1); B=gen(2048,2)
a1=req(A); b=req(B); a2=req(A)
print(f"cache-ram={cr}: A1 wall {a1[0]:.1f}s prompt_n {a1[1]} cache_n {a1[2]} pp {a1[3]}s md5 {a1[4]} | B wall {b[0]:.1f}s | A2 wall {a2[0]:.1f}s prompt_n {a2[1]} cache_n {a2[2]} pp {a2[3]}s md5 {a2[4]}")
PY
  kill -INT $PID; wait $PID 2>/dev/null; sleep 3; done
echo WIN2Q-DONE
