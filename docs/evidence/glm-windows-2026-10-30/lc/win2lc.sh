#!/usr/bin/env bash
# Window lc: Spark GLM-5.3 at 128k context (one slot). Run on the Spark from /srv/glm-5.3 with the production pod down.
# Binary: ~/llama-k3-fdq/build-fdq (production stack + exact set, inert with the exact knobs unset; fdq Q1 = gold).
# Env and args = the glm53-spark manifest (argo-things), except ctx/cache per leg and the port.
# Legs: L64 control (ctx 65536, cache 84000, production): standard x2, must be gold 9449ae5738.
#       L128 (ctx 131072, cache 78000): standard x3, depth sweep 8k/32k/64k/100k, follow-up-turn reuse at 64k.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} != --window ]]; then
  rc=0
  timeout --signal=TERM --kill-after=30s 7000 bash "$SCRIPT_DIR/win2lc.sh" --window || rc=$?
  echo "LC status=$rc"
  echo WIN2LC-DONE
  exit "$rc"
fi
cd /srv/glm-5.3
BUILD=$HOME/llama-k3-fdq/build-fdq
[[ -x $BUILD/bin/llama-server ]] || { echo "ABORT-WINDOW missing $BUILD/bin/llama-server"; exit 1; }
[[ -s snapshot/glm53-q2.k3sn && -r store/glm53-q2-v1.k3xs && -r models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf ]] || { echo "ABORT-WINDOW snapshot/store/model missing"; exit 1; }
mkdir -p legs-lc
[[ -z $(compgen -G 'legs-lc/LC-*') ]] || { echo "ABORT-WINDOW existing legs-lc outputs; preserve them"; exit 1; }
pgrep -f "llama-serve[r] -m " >/dev/null && { echo "ABORT-WINDOW a llama-server is running"; exit 1; }
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_DEBUG_@}; do unset "$v"; done
unset LLAMA_PREFILL_ISOLATE
export LD_LIBRARY_PATH="$BUILD/bin:$BUILD/lib"
pid=
stop_server() {
  [[ -n $pid ]] || return 0
  kill -INT "$pid" 2>/dev/null; for s in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; pid=
}
trap stop_server EXIT
trap 'echo LEG-FAIL window-budget-or-interrupt; exit 1' INT TERM
start() {  # name ctx cache
  local name=$1 ctx=$2 cache=$3
  cp snapshot/glm53-q2.k3sn "legs-lc/$name.k3sn" || return 1
  echo "LEG $name ctx=$ctx cache_MiB=$cache parallel=1 $(date +%T)"
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE="$PWD/store/glm53-q2-v1.k3xs" LLAMA_EXPERT_STORE_SNAPSHOT="$PWD/legs-lc/$name.k3sn" \
    LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 \
    LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 \
    LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 \
    LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1 \
    "$BUILD/bin/llama-server" -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --alias glm-5.3 \
    --host 127.0.0.1 --port 18080 --ctx-size "$ctx" --parallel 1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 \
    --moe-cache-mib "$cache" --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --jinja --no-warmup --fit off --load-mode mmap -lv 4 \
    > "legs-lc/$name.log" 2>&1 &
  pid=$!
  for j in $(seq 1 600); do
    kill -0 "$pid" 2>/dev/null || { echo "LEG-FAIL $name server died"; tail -3 "legs-lc/$name.log"; pid=; return 1; }
    curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $name after ${j}s"; return 0; }
    sleep 1
  done
  echo "LEG-FAIL $name health timeout"; return 1
}
report() {  # name: KV size and memory valve low point from the server log
  local log=legs-lc/$1.log
  grep -a -E "KV buffer size|llama_kv_cache.*MiB" "$log" | head -3 | sed "s/^/KV $1: /"
  grep -a -o "lowest [0-9.]* GB" "$log" | sort -k2 -n | head -1 | sed "s/^/MEMORY $1 MemAvailable /"
  grep -a -E "hit rate" "$log" | tail -1 | cut -c1-200 | sed "s/^/STORE $1: /"
}
run() { python3 -I "$SCRIPT_DIR/lc_client.py" "$@" | tee -a legs-lc/results.jsonl; }

if start LC-64 65536 84000; then
  run std LC-64 2 || echo "LEG-FAIL LC-64 client"
  stop_server; report LC-64
fi
if start LC-128 131072 78000; then
  run std LC-128 3 || echo "LEG-FAIL LC-128 std"
  run depth LC-128 8192,32768,65536,100000 || echo "LEG-FAIL LC-128 depth"
  run reuse LC-128 65536 || echo "LEG-FAIL LC-128 reuse"
  stop_server; report LC-128
fi
python3 - <<'PY'
import json
gold = "9449ae5738f2e56f99ab7ab6e9626eb5"
for l in open("legs-lc/results.jsonl"):
    d = json.loads(l)
    tag = ""
    if d["kind"].startswith("std"):
        tag = "GOLD" if d["md5"] == gold else "NOT-GOLD"
    print(f"SUM {d['leg']} {d['kind']:<15} prompt={d['prompt_n']:>6} cache_n={d['cache_n']} prefill={d['prefill_tps']:>7} tok/s ({d['prefill_s']} s) decode={d['decode_tps']} tok/s md5={d['md5'][:10]} {tag}")
PY
