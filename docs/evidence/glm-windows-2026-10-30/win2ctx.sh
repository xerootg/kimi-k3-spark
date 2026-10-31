#!/bin/bash
# WIN2CTX: production context sweep; one server per ubatch, frozen base copy per leg.
# Full schedule (two passes also in LONG2): 282624 / 60 = 78.5 min prefill alone.
# Budget reduction: omit 24576 and second 32768 pass in BOTH legs.
# Retained: 167936 / 60 = 46.6 min prefill, 11 x 64 decode tokens (~3 min).
# A 74-minute work deadline leaves cleanup margin within the 75-minute window.
set -o pipefail
cd ~/win2 || exit 1
W=~/win2
M=/srv/glm-5.3/models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf
E=/srv/glm-5.3/store/glm53-q2-v1.k3xs
SECONDS=0
PID=
status=0
summary="$W/WIN2CTX-$RANDOM-$$.summary"
: > "$summary" || exit 1
cleanup() {
  if [[ -n "$PID" ]]; then
    kill -INT "$PID" 2>/dev/null
    for ((i=0; i<10; i++)); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
      kill -TERM "$PID" 2>/dev/null
      sleep 1
      kill -KILL "$PID" 2>/dev/null
    fi
    wait "$PID" 2>/dev/null
    PID=
  fi
}
trap cleanup EXIT
trap 'echo ABORT-WINDOW interrupted; exit 130' INT TERM
for leg in LONG1 LONG2; do
  ubatch=2048
  [[ "$leg" == LONG2 ]] && ubatch=512
  if (( SECONDS >= 4440 )); then echo 'ABORT-WINDOW deadline'; status=1; break; fi
  tag=WIN2CTX-$leg-$RANDOM-$$
  cp "$W/base.k3sn" "$W/$tag.k3sn" || { status=1; break; }
  # Refuse an occupied port so health/completion cannot target another owner's server.
  python3 - <<'PY'
import socket
with socket.socket() as s:
    s.bind(('127.0.0.1', 18080))
PY
  if (( $? != 0 )); then echo 'ABORT-WINDOW port 18080 occupied'; status=1; break; fi
  echo "$leg START ubatch=$ubatch snapshot=$W/$tag.k3sn"
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$E LLAMA_EXPERT_STORE_SNAPSHOT=$W/$tag.k3sn LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 \
    ~/llama-k3-bs/build/bin/llama-server -m $M --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size "$ubatch" -ngl 999 --n-cpu-moe 999 --moe-cache-mib 80000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 > "$W/$tag.log" 2>&1 & PID=$!
  ready=0
  while (( SECONDS < 4440 )); do
    if ! kill -0 "$PID" 2>/dev/null; then echo "$leg died"; break; fi
    if curl -s -m 2 127.0.0.1:18080/health | grep -q ok; then ready=1; break; fi
    sleep 3
  done
  if (( ! ready )); then echo "ABORT-WINDOW $leg startup"; status=1; cleanup; break; fi
  remaining=$((4440 - SECONDS))
  if (( remaining <= 0 )); then echo 'ABORT-WINDOW deadline'; status=1; cleanup; break; fi
  timeout --signal=TERM --kill-after=2 "$remaining" python3 -u - "$leg" "$W/$tag.log" "$summary" <<'PY'
import hashlib, json, random, re, sys, time, urllib.request
from pathlib import Path
leg, logname, summary = sys.argv[1:]
log = Path(logname)
pattern = re.compile(r'fill_late|expert store|hit rate|valve|snapshot|cache update|slot|KV|kv|DSA|indexer')
def gen(n, seed):
    r=random.Random(seed); lines=[]; k=0
    while sum(len(l)+1 for l in lines)<n*4:
        lines.append(f"uint64_t f{k}(uint64_t x) {{ return ((x ^ 0x{r.getrandbits(32):08x}ULL) + {k}) >> {k%13}; }}"); k+=1
    return '\n'.join(lines)
def post(endpoint, payload):
    body=json.dumps(payload).encode()
    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18080/'+endpoint, body,
                                {'Content-Type':'application/json'}), timeout=4400) as response:
        return json.load(response)
def tokens(n):
    # Same generator as win2q; tokenize first, then retain exactly N token IDs.
    size=n
    while True:
        ids=post('tokenize', {'content':gen(size, 1), 'add_special':False})['tokens']
        if len(ids)>=n:
            ids=ids[:n]
            if not all(isinstance(t, int) for t in ids):
                raise ValueError('tokenize did not return integer token IDs')
            return ids
        size*=2
schedule=([(2048,1),(2048,2),(8192,1),(8192,2),(16384,1),(16384,2),
           (32768,1),(16384,3)] if leg=='LONG1' else [(16384,1),(16384,2),(32768,1)])
prompts={}
for n, run in schedule:
    if n not in prompts: prompts[n]=tokens(n)
    # Count lines, not bytes; retain the final 12 matching lines emitted by this pass.
    with log.open(errors='replace') as f: first_line=sum(1 for _ in f)
    print(f'{leg} N={n} pass={run} START utc={time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())} log_line={first_line+1}')
    start=time.perf_counter()
    try:
        d=post('completion', {'prompt':prompts[n], 'cache_prompt':False,
                             'n_predict':64, 'temperature':0, 'seed':1})
        wall=(time.perf_counter()-start)*1000
        tm=d['timings']; pp=float(tm['prompt_ms']); dec=float(tm['predicted_ms'])
        pn=int(tm['prompt_n']); dn=int(tm['predicted_n'])
        pps=pn*1000/pp if pp>0 else 0
        ds=dn*1000/dec if dec>0 else 0
        print(f'{leg} N={n} pass={run} prompt_n={pn} prompt_ms={pp:.2f} pp={pps:.2f} tok/s '
              f'predicted_n={dn} predicted_ms={dec:.2f} decode={ds:.2f} tok/s '
              f'md5={hashlib.md5(d["content"].encode()).hexdigest()} wall_ms={wall:.2f} '
              f'unaccounted_ms={wall-pp-dec:.2f} END utc={time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}')
        with open(summary,'a') as f:
            f.write(f'CTX {n}: pp={pps:.2f} decode={ds:.2f} tok/s leg={leg} pass={run}\n')
        if pn != n or dn != 64:
            raise RuntimeError(f'incomplete measurement: prompt_n={pn}, predicted_n={dn}, expected {n}/64')
    finally:
        with log.open(errors='replace') as f:
            matches=[line.rstrip('\n') for i,line in enumerate(f) if i>=first_line and pattern.search(line)]
        print(f'{leg} N={n} pass={run} LOG lines_after={first_line} matches={len(matches)} showing_last={min(12,len(matches))}')
        for line in matches[-12:]: print(line[:200])
PY
  py_status=$?
  cleanup
  if (( py_status != 0 )); then echo "ABORT-WINDOW $leg measurement status=$py_status"; status=$py_status; break; fi
done
cat "$summary"
echo WIN2CTX-DONE
exit "$status"
