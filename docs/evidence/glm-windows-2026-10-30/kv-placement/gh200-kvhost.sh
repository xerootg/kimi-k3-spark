#!/bin/bash
# Patch 0051 on the GH200: GLM-5.3-Flash Q4's KV + indexer cache in HBM (production: elastic KV) vs in Grace memory
# read over NVLink-C2C (LLAMA_KV_HOST=1), same binary, production env and args otherwise (minus --mmproj).
# Prepared on the Spark 2026-11-08 (kimi-k3-spark docs/experiments.md, "KV in host memory, patch 0051").
#
# Bundle: copy this directory to the GH200 first, e.g. from a kimi-k3-spark checkout:
#   ssh ubuntu@192.222.50.246 mkdir -p /srv/glm-5.3-flash/kh
#   scp -r patches docs/evidence/glm-windows-2026-10-30/kv-placement/{gh200-kvhost.sh,kvprobe.cu,bench.py,needle.py,opprof.py} ubuntu@192.222.50.246:/srv/glm-5.3-flash/kh/
#
# 1) BEFORE the downtime (production keeps running, ~10-15 min, niced):   ./gh200-kvhost.sh build
#    upstream llama.cpp at LLAMA_SHA + every patch in ./patches (0001..0051), sm_90, into ~/llama-kh/build;
#    kvprobe for sm_90. Fallback if the build fails: the CI image of the commit that added 0051
#    (ghcr.io/xerootg/llamacpp-spark:547208dca250-g<sha>, sm_90 + sm_121) holds the same binary.
# 2) CLAIM in the coordination scratchpad, glm53-gh200 to 0 replicas, then (detached, under the window lock):
#    setsid nohup ./gh200-kvhost.sh probe HBM-perf GRACE-perf HBM-prof GRACE-prof > window.log 2>&1 < /dev/null &
#    About 60 min: probe ~3, each perf leg ~14 (start ~1, standard request 1+3 rounds, 16k / 100k / 500k prompts
#    with 128 decode tokens), each prof leg ~15 (100k and 500k prompts, 64 decode tokens, op profiler on, then a
#    needle: a fact at the start of a 100k prompt, the question at the end).
#    Optional extra leg GRACETHP-perf (LLAMA_KV_HOST_THP=1, 512 MiB THPs) if the probe shows "reg" beating
#    "regnothp" by a wide margin.
# 3) Scale glm53-gh200 back to 1, verify through the Service (standard md5 2a73cda2a3), RELEASE.
#
# Expected on the GH200 (window flashwin2, same body as bench.py conc): standard request md5 9c146a59b5 (256 tokens,
# ~48-52 tok/s); the Service check's 2a73cda2a3 is a different request.
# What to compare: decode tok/s at each depth (HBM vs Grace), md5 equal at every depth, and in the prof legs the
# FLASH_ATTN_EXT, GET_ROWS kv=cache_idx_k (gather_pooled) and LIGHTNING_INDEXER rows of the last op profile before
# each request ends (ms/graph). LOGITS_FNV lines must be identical between the two prof legs.
set -u
LLAMA_SHA=547208dca250ceb3e4c8615598d043f2b602a527
KH=$(cd "$(dirname "$0")" && pwd)
SRV=/srv/glm-5.3-flash
SRC=$HOME/llama-kh; BD=$SRC/build; BIN=$BD/bin/llama-server

if [ "${1:-}" = build ]; then
  export PATH=/usr/local/cuda/bin:$PATH
  rm -rf "$SRC" && git init -q "$SRC" && cd "$SRC" && git remote add origin https://github.com/ggml-org/llama.cpp \
    && git fetch -q --depth 1 origin $LLAMA_SHA && git checkout -q FETCH_HEAD || { echo BUILD-FAIL fetch; exit 1; }
  for p in "$KH"/patches/*.patch; do git apply "$p" || { echo "BUILD-FAIL apply $p"; exit 1; }; done
  ls "$KH"/patches/0051-*.patch >/dev/null || { echo "BUILD-FAIL no 0051 in the bundle"; exit 1; }
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$KH/cmake.log" 2>&1 \
    && nice -n 19 cmake --build build -j 32 --target llama-server > "$KH/build.log" 2>&1 || { echo "BUILD-FAIL see $KH/build.log"; exit 1; }
  nvcc -O2 -arch=sm_90 -o "$KH/kvprobe" "$KH/kvprobe.cu" || { echo BUILD-FAIL kvprobe; exit 1; }
  echo "BUILD-OK $BIN"; exit 0
fi

cd $SRV; OUT=$KH/legs; mkdir -p $OUT
exec 9>$SRV/window.lock; flock 9
export LD_LIBRARY_PATH="$BD/bin"
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_ELASTIC_@} ${!LLAMA_KV_@}; do unset "$v"; done
# production env (argo-things configs/glm53-gh200/llama-server.yaml, image g7f117e1) with native paths
PROD_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$SRV/store/glm53flash-q4.k3xs LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=40960 LLAMA_EXPERT_STORE_VALVE_FLOOR=10 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_ELASTIC_KV=1 LLAMA_EXPERT_STORE_HBM_SPARE=1 LLAMA_EXPERT_STORE_MAX_BATCH=4096 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0"
ARGS="-m q4/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --alias glm-5.3-flash --host 127.0.0.1 --port 18104 --ctx-size 1048576 --parallel 1 --batch-size 4096 --ubatch-size 4096 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 189360 --cache-ram 98304 --temp 1.0 --top-p 0.95 --threads 16 --threads-batch 32 --flash-attn on --jinja --metrics --no-warmup --fit off --load-mode mmap -lv 4"

leg() { # leg TAG placement(HBM|GRACE|GRACETHP) mode(perf|prof)
  local TAG=$1 PL=$2 MODE=$3 EXTRA=""
  case $PL in GRACE) EXTRA="LLAMA_KV_HOST=1";; GRACETHP) EXTRA="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1";; esac
  [ $MODE = prof ] && EXTRA="$EXTRA GGML_CUDA_OP_PROFILE=16 LLAMA_LOGITS_FNV=1"
  pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT $TAG: a llama-server is running (is glm53-gh200 at 0?)"; return 1; }
  [ -x "$BIN" ] || { echo "ABORT $TAG: no $BIN (run build first)"; return 1; }
  cp snapshot/glm53flash-q4-full.k3sn $OUT/snap-$TAG.k3sn
  echo "=== LEG $TAG placement=$PL mode=$MODE extra=[$EXTRA] $(date +%T)"
  (while sleep 2; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"; done) > $OUT/mem-$TAG.txt & local MON=$!
  env $PROD_ENV LLAMA_EXPERT_STORE_SNAPSHOT=$OUT/snap-$TAG.k3sn $EXTRA $BIN $ARGS > $OUT/server-$TAG.log 2>&1 & local pid=$!
  ( while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      if [ "$a" -lt 8192 ]; then echo "WATCHDOG kill $TAG ${a} MiB" | tee -a $OUT/watchdog.txt; kill -KILL $pid; break; fi; sleep 1; done ) & local WD=$!
  local ok=0 i
  for i in $(seq 1 200); do
    curl -fsS -m 2 127.0.0.1:18104/health 2>/dev/null | grep -q ok && { echo "READY $TAG after $((i*3))s"; ok=1; break; }
    kill -0 $pid 2>/dev/null || { echo "LEG-FAIL $TAG died"; grep -a -i -E "error|abort|assert|fail" $OUT/server-$TAG.log | tail -5 | cut -c1-300; break; }
    sleep 3
  done
  if [ $ok = 1 ]; then
    grep -a -E "LLAMA_KV_HOST|host_kv_buffer|KV buffer size|elastic KV:" $OUT/server-$TAG.log | cut -c1-260
    if [ $MODE = perf ]; then
      python3 -I $KH/bench.py conc --url http://127.0.0.1:18104 --tag $TAG --n 1 --rounds 3 --n-predict 256 | tee -a $OUT/results.jsonl
      python3 -I $KH/bench.py depth --url http://127.0.0.1:18104 --tag $TAG-depth --lens 16384,100000,500000 --n-predict 128 | tee -a $OUT/results.jsonl
    else
      python3 -I $KH/bench.py depth --url http://127.0.0.1:18104 --tag $TAG-depth --lens 100000,500000 --n-predict 64 | tee -a $OUT/results.jsonl
      python3 -I $KH/needle.py http://127.0.0.1:18104 $TAG-needle 100000 | tee -a $OUT/results.jsonl
    fi
  fi
  kill -INT $pid 2>/dev/null; for i in $(seq 1 90); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null
  kill $WD $MON 2>/dev/null
  [ $MODE = prof ] && { grep -a -o "LOGITS_FNV.*" $OUT/server-$TAG.log > $OUT/$TAG.fnv; echo "fnv lines $(wc -l < $OUT/$TAG.fnv)"; }
  echo "PEAK GPU $(awk '{print $2}' $OUT/mem-$TAG.txt | sort -n | tail -1) MiB, LOWEST MemAvailable $(awk '{print $3}' $OUT/mem-$TAG.txt | sort -n | head -1) MiB; END $TAG $(date +%T)"
  rm -f $OUT/snap-$TAG.k3sn
}

for L in "${@:-probe HBM-perf GRACE-perf HBM-prof GRACE-prof}"; do
  for L1 in $L; do
    case $L1 in
      probe) pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT probe: a llama-server is running"; continue; }
             echo "=== kvprobe $(date +%T)"; $KH/kvprobe 24 16384,100000,500000,1000000 dev,reg,regnothp,halloc 2>&1 | tee $OUT/kvprobe.txt ;;
      HBM-perf|GRACE-perf|GRACETHP-perf|HBM-prof|GRACE-prof|GRACETHP-prof) leg ${L1/-/} ${L1%%-*} ${L1##*-} ;;
      *) echo "unknown leg $L1" ;;
    esac
  done
done
pgrep -f "[l]lama-server -m " >/dev/null && echo "WARN a llama-server is still running" || echo "no test server left"
# summary: decode tok/s and md5 per leg and depth; md5 must match between HBM and GRACE at every depth
python3 -I - "$OUT/results.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for r in rows:
    if r["kind"] == "needle":
        print(f'{r["tag"]:>14} needle {r["prompt_n"]:>7}  found {r["found"]}  md5 {r["md5"]}')
    elif r["kind"] == "conc":
        print(f'{r["tag"]:>14} std   round {r["round"]!s:>4}  {r["per_stream_tps"][0]:6.2f} tok/s  md5 {r["md5"][0]}')
    else:
        print(f'{r["tag"]:>14} depth {r["prompt_n"]:>8}  prefill {r["prefill_tps"]:7.1f}  decode {r["decode_tps"]:6.2f} tok/s  md5 {r["md5"]}')
PY
[ -f $OUT/server-HBMprof.log ] && [ -f $OUT/server-GRACEprof.log ] && python3 -I $KH/opprof.py $OUT/server-HBMprof.log $OUT/server-GRACEprof.log --labels 100k,500k,needle
[ -f $OUT/HBMprof.fnv ] && [ -f $OUT/GRACEprof.fnv ] && { cmp -s $OUT/HBMprof.fnv $OUT/GRACEprof.fnv && echo "LOGITS_FNV identical ($(wc -l < $OUT/HBMprof.fnv) lines)" || echo "LOGITS_FNV DIFFER"; }
echo "GH200-KVHOST-DONE $(date +%T)"
