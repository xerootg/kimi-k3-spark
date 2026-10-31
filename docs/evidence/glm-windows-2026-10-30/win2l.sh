#!/bin/bash
# Spark window l: copied c stack, frozen snapshot, LFU v2 A/B.
set -euo pipefail
abort() { echo "ABORT-WINDOW $*"; exit 1; }
trap 'abort "line $LINENO failed"' ERR
pid=
stop_server() {
  if [[ -n $pid ]]; then
    kill -INT "$pid" 2>/dev/null || :
    for ((s=0; s<15; s++)); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || :; fi
    wait "$pid" 2>/dev/null || :
    pid=
  fi
}
trap stop_server EXIT
trap 'abort interrupted' INT TERM
cd /srv/glm-5.3
SRC=$HOME/llama-k3-c
DST=$HOME/llama-k3-l
PATCH=$HOME/win2/0018-lfu-policy.v2.patch
[[ ! -e $DST && ! -L $DST ]] || abort '~/llama-k3-l exists'
[[ -x $SRC/build/bin/llama-server && -s $SRC/build/CMakeCache.txt ]] || abort 'c tree has no native build/bin/llama-server or CMakeCache'
[[ -s $PATCH ]] || abort 'missing ~/win2/0018-lfu-policy.v2.patch'
[[ -s snapshot/glm53-q2.k3sn && -r store/glm53-q2-v1.k3xs && -r models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf ]] || abort 'snapshot/store/model missing'
python3 - <<'PY' || abort 'existing llama-server -m process or occupied port 18080'
import pathlib,socket
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try: args=p.read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError): continue
    assert not (args and b'llama-server' in args[0] and b'-m' in args), str(p)
with socket.socket() as s: s.bind(('127.0.0.1',18080))
PY
mkdir -p legs
for name in LF0 LF1 LF0b LF1b; do
  [[ ! -e legs/$name.log && ! -e legs/$name.k3sn ]] || abort "existing $name outputs; archive first"
done
cp -a "$SRC" "$DST"
git -C "$DST" apply --check "$PATCH" || abort '0018 v2 apply-check failed'
git -C "$DST" apply "$PATCH" || abort '0018 v2 apply failed'
# reviewer's P1 fix (speculative headroom counted victims in protected LFU layers): unconditional, both legs carry it
git -C "$DST" apply --check ~/win2/0018-lfu-headroom.patch && git -C "$DST" apply ~/win2/0018-lfu-headroom.patch || abort '0018 headroom fix does not apply'
# reviewer's P2 fix: no heap growth of the protected-key buffers on the hook path
git -C "$DST" apply --check ~/win2/0018-lfu-nogrow.patch && git -C "$DST" apply ~/win2/0018-lfu-nogrow.patch || abort '0018 nogrow fix does not apply'
export CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH
python3 - "$SRC" "$DST" <<'PY'
import pathlib,sys
src,dst=sys.argv[1:]
for p in (pathlib.Path(dst)/'build').rglob('*'):
    if p.is_file() and not p.is_symlink() and (p.name in ('CMakeCache.txt','Makefile','build.ninja','rules.ninja') or p.suffix in ('.cmake','.make')):
        data=p.read_bytes()
        if src.encode() in data: p.write_bytes(data.replace(src.encode(),dst.encode()))
PY
if (cd "$DST" && cmake -S . -B build && nice -n 19 cmake --build build -j 12 --target llama-server) > "$HOME/win2/build18b.log" 2>&1; then
  build18b=0
else
  build18b=$?
fi
echo "build18b=$build18b"
((build18b == 0)) || abort 'configure/build failed (~/win2/build18b.log)'
BIN=$DST/build/bin/llama-server
[[ -x $BIN && $BIN -nt $PATCH ]] || abort 'binary missing or not newer than patch'
cp snapshot/glm53-q2.k3sn legs/L-base.k3sn
P='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":128,"cache_prompt":false}'
# Keep inherited remote experiments out of this local-bank window.
for v in ${!LLAMA_EXPERT_STORE_REMOTE@}; do unset "$v"; done
unset LLAMA_EXPERT_STORE_POLICY LLAMA_EXPERT_STORE_LFU_HALF_LIFE
STORE=glm53-q2-v1.k3xs
SZ=1 HL=1
leg() { local name=$1 lfu=$2 i j ready=0 failure=
  local -a policy_env=()
  if ((lfu)); then
    policy_env=(LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144)
  fi
  cp legs/L-base.k3sn "legs/$name.k3sn"
  echo "$name LFU=$lfu SZ=$SZ HL=$HL PREFETCH_WIDE=0"
  env "${policy_env[@]}" LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/$STORE LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/legs/$name.k3sn \
  LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 \
  LLAMA_EXPERT_STORE_RESERVE_MIB=4096 ${PROF:+GGML_CUDA_OP_PROFILE=$PROF} LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=$SZ LLAMA_DSA_SHORT_CTX_SKIP=0 LLAMA_EXPERT_STORE_HOOK_LEAN=$HL LLAMA_EXPERT_STORE_HOOK_STATS=1 \
  LLAMA_EXPERT_STORE_PREFETCH_WIDE=0 \
  "$BIN" -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
  --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size 512 -ngl 999 --n-cpu-moe 999 \
  --moe-cache-mib 80000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 \
  > "legs/$name.log" 2>&1 & pid=$!
  for ((j=0; j<1200; j++)); do
    if ! kill -0 "$pid" 2>/dev/null; then failure="$name server died before health"; break; fi
    if curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok; then ready=1; break; fi
    sleep 1
  done
  if [[ -z $failure ]] && (( ! ready )); then failure="$name health timeout"; fi
  if [[ -z $failure ]]; then
    for i in 0 1 2; do
      if ! curl -fsS -m 900 127.0.0.1:18080/completion -H 'Content-Type: application/json' -d "$P" > "legs/$name.r$i.json"; then
        failure="$name request $i failed"; break
      fi
      if ! python3 - "$name" "$i" <<'PYGATE'
import json,hashlib,sys
name,i=sys.argv[1:]
try:
    d=json.load(open(f'legs/{name}.r{i}.json')); t=d['timings']
    h=hashlib.md5(d['content'].encode()).hexdigest()[:10]
    if i!='0': print(name,f'r{i}',round(t['predicted_per_second'],3),'tok/s md5',h,flush=True)
    assert t['predicted_n']==128 and h=='9449ae5738', 'gold mismatch'
except Exception as e:
    print(f'{name} response gate r{i}: {e}', flush=True)
    sys.exit(1)
PYGATE
      then
        failure="$name response/md5 gate failed r$i"; break
      fi
    done
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    failure="${failure:+$failure; }$name server died at end of leg"
  fi
  stop_server
  grep -a -E 'host bank hooks|host bank real (readiness|bucket|histogram)|split GEMV:|hit rate|expert store:.*lookups|expert store: LFU eviction|LFU (blocked:|window |selected:|slot )' "legs/$name.log" | cut -c1-220 || :
  if ((lfu)); then
    if [[ -z $failure ]]; then echo "LFU-OK $name"; else echo "LFU-FAIL $failure"; fi
  elif [[ -n $failure ]]; then
    abort "$failure"
  fi
  echo "$name MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
  sleep 5
}
leg LF0 0
leg LF1 1
leg LF0b 0
leg LF1b 1
echo WIN2L-DONE
