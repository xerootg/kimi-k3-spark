#!/bin/bash
# GLM-5.3-Flash UD-Q4_K_XL on the GH200, 1M context, elastic KV, spare slots. glm53-gh200 must be at 0.
# Phase 0: seed run (cold, 2k prompt, SIGINT) -> full snapshot. Phase 1: U=4096 tier-0 probe (HBM left after load).
# Phase 2: legs U=2048 and U=4096 with the largest HBM tier that leaves the reserve plus 4 GiB.
set -u
cd /srv/glm-5.3-flash; out=win; mkdir -p $out
BIN=$HOME/llama-el/build/bin/llama-server
M=q4/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT a llama-server is running"; exit 1; }
STORE_GB=$(( $(stat -c %s store/glm53flash-q4.k3xs) / 1000000000 ))
CACHE_MIB=$(( STORE_GB * 1000 * 1000 / 1048 * 105 / 100 ))   # store size + 5 % (MiB)
echo "store ${STORE_GB} GB, cache ${CACHE_MIB} MiB"
BASE="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3-flash/store/glm53flash-q4.k3xs LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_SPARE=1 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0 LLAMA_ELASTIC_KV=1"
pid=
start() { # tag U hbm_mib snapshot [extra args...]
  local tag=$1 U=$2 H=$3 SN=$4; shift 4
  echo "=== $tag U=$U hbm=$H $(date +%T)"
  (while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done) > $out/gpumem-$tag.txt & mon=$!
  env $BASE LLAMA_EXPERT_STORE_HBM_MIB=$H LLAMA_EXPERT_STORE_MAX_BATCH=$U LLAMA_EXPERT_STORE_SNAPSHOT=$SN "$BIN" -m $M --alias glm-5.3-flash --host 127.0.0.1 --port 18104 \
    --ctx-size 1048576 --parallel 1 --batch-size $U --ubatch-size $U --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib $CACHE_MIB --cache-ram 0 \
    --threads 16 --threads-batch 32 --flash-attn on --jinja --no-warmup --fit off --load-mode mmap -lv 4 "$@" > $out/server-$tag.log 2>&1 & pid=$!
  local t0=$(date +%s)
  for i in $(seq 1 300); do curl -s -m 2 http://127.0.0.1:18104/health | grep -q ok && { echo "READY $tag $(( $(date +%s) - t0 )) s"; grep -a -E "slots \(|snapshot restored|compute buffer size|model buffer size|KV buffer|elastic KV:|released [0-9]+ slots" $out/server-$tag.log | cut -c1-200 | head -16; return 0; }; kill -0 $pid 2>/dev/null || { echo "FAIL $tag"; grep -a -i -E "error|abort|out of memory|failed" $out/server-$tag.log | tail -4 | cut -c1-220; kill $mon; pid=; return 1; }; sleep 3; done
  echo "FAIL $tag timeout"; return 1
}
stop() { [ -n "$pid" ] || return 0; kill -INT $pid; for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null; pid=; kill $mon 2>/dev/null; echo "PEAK $(sort -n $out/gpumem-*.txt | tail -1) MiB (window so far)"; }
# phase 0: seed
if [ ! -s snapshot/glm53flash-q4-full.k3sn ]; then
  rm -f $out/seed.k3sn
  if start seed 2048 0 /srv/glm-5.3-flash/$out/seed.k3sn; then
    python3 -I - <<PY
import json,urllib.request,random
W="kubernetes pod node volume garage traefik ingress service deployment".split(); r=random.Random(3)
p="Summarize:\n"+" ".join(r.choice(W) for _ in range(1600))
d=json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18104/completion",data=json.dumps({"prompt":p,"n_predict":8,"temperature":0}).encode(),headers={"Content-Type":"application/json"}),timeout=900).read())
print("seed prompt", d["timings"]["prompt_n"], "tokens")
PY
    stop
  fi
  python3 -I /srv/glm-5.3/full-snapshot.py store/glm53flash-q4.k3xs $out/seed.k3sn snapshot/glm53flash-q4-full.k3sn && echo "FULL-SNAPSHOT OK $(stat -c %s snapshot/glm53flash-q4-full.k3sn) B" || { echo "FULL-SNAPSHOT FAIL"; exit 1; }
fi
# phase 1: probe
cp snapshot/glm53flash-q4-full.k3sn $out/snap-probe.k3sn
if start probe 4096 0 /srv/glm-5.3-flash/$out/snap-probe.k3sn; then
  sleep 5; USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); TOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  echo "PROBE U=4096 tier 0: GPU used $USED of $TOT MiB"; stop
fi
FREE4=$(( TOT - USED ))
# compute buffer grows ~linearly with U: assume U=2048 needs half the U=4096 CUDA0 compute buffer
CB4=$(grep -a "CUDA0 compute buffer size" $out/server-probe.log | tail -1 | sed -E "s/.*= *([0-9]+).*/\1/")
H4=$(( (FREE4 - 8192 - 4096) / 1024 * 1024 )); H2=$(( (FREE4 + CB4/2 - 8192 - 4096) / 1024 * 1024 ))
[ $H4 -lt 0 ] && H4=0; echo "tiers: U=4096 -> $H4 MiB, U=2048 -> $H2 MiB (CUDA0 compute at 4096: $CB4 MiB)"
# phase 2
for leg in "u4096 4096 $H4" "u2048 2048 $H2"; do
  set -- $leg; cp snapshot/glm53flash-q4-full.k3sn $out/snap-$1.k3sn
  if start $1 $2 $3 /srv/glm-5.3-flash/$out/snap-$1.k3sn; then python3 -I flash_client.py $1 | tee -a $out/results.jsonl; grep -a "hit rate" $out/server-$1.log | tail -1 | cut -c1-160; stop; fi
  rm -f $out/snap-$1.k3sn
done
# MTP leg left for a later window (an MTP server needs its own snapshot)
rm -f $out/snap-probe.k3sn
pgrep -f "[l]lama-server -m " && echo "WARN server still running" || echo "no server running"
echo FLASHWIN-DONE
