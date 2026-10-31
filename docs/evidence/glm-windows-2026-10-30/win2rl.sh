#!/bin/bash
# Spark window rl: direct-read floor, then early-record latency on the frozen c stack.
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
BENCH=$HOME/win2/tools/gb10-readlat-bench.cpp
[[ -r $BENCH ]] || abort 'missing ~/win2/tools/gb10-readlat-bench.cpp'
g++ -O2 -std=c++17 -pthread "$BENCH" -o "$HOME/win2/readlat-bench"
echo 'readlat backend=pread'
"$HOME/win2/readlat-bench" /srv/glm-5.3/store/glm53-q2-v1.k3xs
if printf '#include <liburing.h>\nint main(){io_uring r; return io_uring_queue_init(8,&r,0); }\n' | g++ -x c++ - -luring -o "$HOME/win2/readlat-uring-probe" 2>/dev/null; then
  g++ -O2 -std=c++17 -pthread -DUSE_URING "$BENCH" -luring -o "$HOME/win2/readlat-bench-uring"
  echo 'readlat backend=io_uring'
  "$HOME/win2/readlat-bench-uring" /srv/glm-5.3/store/glm53-q2-v1.k3xs
else
  echo 'readlat io_uring unavailable; pread table above'
fi
SRC=$HOME/llama-k3-c
DST=$HOME/llama-k3-rl
PATCH=$HOME/win2/0012d-early-read-latency.patch
[[ ! -e $DST && ! -L $DST ]] || abort '~/llama-k3-rl exists'
[[ -x $SRC/build/bin/llama-server && -s $SRC/build/CMakeCache.txt ]] || abort 'c tree has no native build/bin/llama-server or CMakeCache'
[[ -s $PATCH ]] || abort 'missing ~/win2/0012d-early-read-latency.patch'
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
for name in RL0 RL0b; do
  [[ ! -e legs/$name.log && ! -e legs/$name.k3sn ]] || abort "existing $name outputs; archive first"
done
cp -a "$SRC" "$DST"
git -C "$DST" apply --check "$PATCH" || abort '0012d apply-check failed'
git -C "$DST" apply "$PATCH" || abort '0012d apply failed'
export CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH
python3 - "$SRC" "$DST" <<'PY'
import pathlib,sys
src,dst=sys.argv[1:]
for p in (pathlib.Path(dst)/'build').rglob('*'):
    if p.is_file() and not p.is_symlink() and (p.name in ('CMakeCache.txt','Makefile','build.ninja','rules.ninja') or p.suffix in ('.cmake','.make')):
        data=p.read_bytes()
        if src.encode() in data: p.write_bytes(data.replace(src.encode(),dst.encode()))
PY
if (cd "$DST" && cmake -S . -B build && nice -n 19 cmake --build build -j 12 --target llama-server) > "$HOME/win2/build12d.log" 2>&1; then
  build12d=0
else
  build12d=$?
fi
echo "build12d=$build12d"
((build12d == 0)) || abort 'configure/build failed (~/win2/build12d.log)'
BIN=$DST/build/bin/llama-server
[[ -x $BIN && $BIN -nt $PATCH ]] || abort 'binary missing or not newer than patch'
unset LLAMA_EXPERT_STORE_LATE_WAIT_IN_KERNEL LLAMA_EXPERT_STORE_AFFINITY LLAMA_EXPERT_STORE_POLLER_SCHED LLAMA_EXPERT_STORE_CPUS LLAMA_EXPERT_STORE_DEMAND_THREADS
cp snapshot/glm53-q2.k3sn legs/RL-base.k3sn
P='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":128,"cache_prompt":false}'
# Keep inherited remote experiments out of this local-bank window.
for v in ${!LLAMA_EXPERT_STORE_REMOTE@}; do unset "$v"; done
STORE=glm53-q2-v1.k3xs
SZ=1 HL=1
leg() { local name=$1 i ready=0
  cp legs/RL-base.k3sn "legs/$name.k3sn"
  echo "$name SZ=$SZ HL=$HL GPU_SLOTS=1 ASYNC=1 PREFETCH_WIDE=0"
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/$STORE LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/legs/$name.k3sn \
  LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 \
  LLAMA_EXPERT_STORE_RESERVE_MIB=4096 ${PROF:+GGML_CUDA_OP_PROFILE=$PROF} LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=$SZ LLAMA_DSA_SHORT_CTX_SKIP=0 LLAMA_EXPERT_STORE_HOOK_LEAN=$HL LLAMA_EXPERT_STORE_HOOK_STATS=1 \
  LLAMA_EXPERT_STORE_GPU_SLOTS=1 GGML_CUDA_HOST_BANK_ASYNC=1 LLAMA_EXPERT_STORE_PREFETCH_WIDE=0 \
  "$BIN" -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
  --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size 512 -ngl 999 --n-cpu-moe 999 \
  --moe-cache-mib 80000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 \
  > "legs/$name.log" 2>&1 & pid=$!
  for ((j=0; j<1200; j++)); do
    kill -0 "$pid" 2>/dev/null || abort "$name server died"
    if curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok; then ready=1; break; fi
    sleep 1
  done
  ((ready)) || abort "$name health timeout"
  for i in 0 1 2; do
    curl -fsS -m 900 127.0.0.1:18080/completion -H 'Content-Type: application/json' -d "$P" > "legs/$name.r$i.json" || abort "$name request $i failed"
    python3 - "$name" "$i" <<'PY' || abort "$name response/md5 gate failed"
import json,hashlib,sys
name,i=sys.argv[1:]
d=json.load(open(f'legs/{name}.r{i}.json')); t=d['timings']
h=hashlib.md5(d['content'].encode()).hexdigest()[:10]
if i!='0': print(name,f'r{i}',round(t['predicted_per_second'],3),'tok/s md5',h,flush=True)
assert t['predicted_n']==128 and h=='9449ae5738', 'gold mismatch'
PY
  done
  stop_server
  grep -a -E 'expert store: demand reader|host bank hooks|host bank real (readiness|bucket|histogram)|host bank c_real issued|host bank record|host bank physical|split GEMV:|hit rate|GB read for decode' "legs/$name.log" || :
  grep -a -q 'host bank hooks' "legs/$name.log" || abort "$name missing hook stats"
  echo "$name MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
  sleep 5
}
leg RL0
rl0_status=$?
leg RL0b
rl0b_status=$?
if ((rl0_status == 0 && rl0b_status == 0)); then
  echo WIN2RL-DONE
else
  abort "RL legs failed: RL0=$rl0_status RL0b=$rl0b_status"
fi
