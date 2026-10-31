#!/bin/bash
# Spark window o: apply 0021 only in a copy; frozen snapshot per server leg.
set -euo pipefail
# Includes copying, builds and checks; TERM cleanup plus 20 s grace stays <45 min.
if [[ ${1:-} != --window-budget ]]; then
  exec timeout --signal=TERM --kill-after=20s 2670s bash "$0" --window-budget
fi
abort() { echo "ABORT-WINDOW $*"; exit 1; }
trap 'abort "line $LINENO failed"' ERR
pid=
stop_server() {
  if [[ -n $pid ]]; then
    kill -INT "$pid" 2>/dev/null || :
    for ((s=0; s<10; s++)); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || :; fi
    wait "$pid" 2>/dev/null || :
    pid=
  fi
}
trap stop_server EXIT
trap 'abort "interrupted or 44.5-minute budget exhausted"' INT TERM
cd /srv/glm-5.3
W=$HOME/win2
SRC=$HOME/llama-k3-c
DST=$HOME/llama-k3-o
PATCH=$W/0021-prefill-order-independent.patch
[[ ! -e $DST && ! -L $DST ]] || abort 'target ~/llama-k3-o exists'
[[ -x $SRC/build/bin/llama-server && -s $SRC/build/CMakeCache.txt ]] || abort 'source build/bin/llama-server or CMake cache missing'
[[ -s $PATCH ]] || abort "missing $PATCH"
for f in mmid_order_check.cpp mmid_order_check.sh; do
  [[ -s $W/tools/validation/$f ]] || abort "missing $W/tools/validation/$f"
done
[[ -s snapshot/glm53-q2.k3sn ]] || abort 'missing frozen snapshot'
[[ -r store/glm53-q2-v1.k3xs && -r models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf ]] || abort 'missing store/model'
mkdir -p legs
[[ ! -e legs/O-window ]] || abort 'legs/O-window exists; archive before rerunning'
mkdir legs/O-window
OUT=/srv/glm-5.3/legs/O-window
python3 - <<'PY' || abort 'existing llama-server -m process or occupied port 18080'
import pathlib,socket
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try: args=p.read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError): continue
    assert not (args and b'llama-server' in args[0] and b'-m' in args), str(p)
with socket.socket() as s: s.bind(('127.0.0.1',18080))
PY
cp snapshot/glm53-q2.k3sn "$OUT/base.k3sn"
cp -a "$SRC" "$DST"
if ! git -C "$DST" apply --check "$PATCH" > "$OUT/apply-check.log" 2>&1; then
  cat "$OUT/apply-check.log"
  abort '0021 git apply --check failed on copy'
fi
git -C "$DST" apply "$PATCH"
export CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH
# Owner window i pattern: rewrite copied CMake absolute paths before reconfigure.
python3 - "$SRC" "$DST" <<'PY'
import pathlib,sys
src,dst=sys.argv[1:]
b=pathlib.Path(dst)/'build'
for p in b.rglob('*'):
    if p.is_file() and not p.is_symlink() and (p.name in ('CMakeCache.txt','Makefile','build.ninja','rules.ninja') or p.suffix in ('.cmake','.make')):
        data=p.read_bytes()
        if src.encode() in data: p.write_bytes(data.replace(src.encode(),dst.encode()))
PY
# The validation driver is standalone g++, not a CMake target. Its script rebuilds
# it for each backend in isolated working directories, never in the stock tree.
mkdir -p "$OUT/stock/tools/validation/prefill-order" "$OUT/patched/tools/validation/prefill-order"
for kind in stock patched; do
  cp "$W/tools/validation/mmid_order_check.cpp" "$W/tools/validation/mmid_order_check.sh" "$OUT/$kind/tools/validation/"
done
if (cd "$DST" && cmake -S . -B build && nice -n 19 cmake --build build -j 12 --target llama-server &&
    g++ -O2 -std=c++17 -I"$DST/ggml/include" "$OUT/patched/tools/validation/mmid_order_check.cpp" \
      -L"$DST/build/bin" -Wl,-rpath,"$DST/build/bin" -lggml-cuda -lggml -lggml-base \
      -o "$OUT/patched/tools/validation/prefill-order/mmid_order_check") > "$W/build21.log" 2>&1; then
  build_rc=0
else
  build_rc=$?
fi
echo "build21=$build_rc"
((build_rc==0)) || abort '0021 configure/build failed; see ~/win2/build21.log'
order_check() {
  local kind=$1 tree=$2 rc
  if (cd "$OUT/$kind" && L="$tree" B="$tree/build" bash tools/validation/mmid_order_check.sh) > "$OUT/$kind/check.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  echo "ORDER-CHECK $kind last: $(tail -1 "$OUT/$kind/check.log")"
  if ((rc==0)); then
    printf -v "$kind" '%s' PASS
  elif grep -q ' differ:' "$OUT/$kind/check.log"; then
    printf -v "$kind" '%s' FAIL
  else
    tail -10 "$OUT/$kind/check.log"
    abort "$kind order-check build/runtime failure rc=$rc"
  fi
}
stock=FAIL patched=FAIL
order_check stock "$SRC"
order_check patched "$DST"
echo "ORDER-CHECK stock=$stock patched=$patched"
if [[ $stock == PASS ]]; then
  echo 'ORDER-CHECK stock PASS: synthetic shapes did not exercise the split; model legs decide (split coverage is not instrumented).'
fi
# Keep all stock token-1 bytes. A change here would contradict the decode bypass.
if [[ $patched == PASS ]]; then
  for f in "$OUT/stock/tools/validation/prefill-order/"t1-*.raw "$OUT/stock/tools/validation/prefill-order/"t1-*.sum; do
    cmp "$f" "$OUT/patched/tools/validation/prefill-order/${f##*/}" || abort 'stock/patched token-1 bytes differ'
  done
fi
for v in ${!LLAMA_EXPERT_STORE_REMOTE@}; do unset "$v"; done
SZ=1 HL=1
leg() {
  local name=$1 BIN=$2 mode=$3 ready=0
  cp "$OUT/base.k3sn" "$OUT/$name.k3sn"
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT="$OUT/$name.k3sn" \
  LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 \
  LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 \
  LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=$SZ LLAMA_DSA_SHORT_CTX_SKIP=0 LLAMA_EXPERT_STORE_HOOK_LEAN=$HL LLAMA_EXPERT_STORE_HOOK_STATS=1 \
  "$BIN" -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
  --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size 512 -ngl 999 --n-cpu-moe 999 \
  --moe-cache-mib 80000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 \
  > "$OUT/$name.log" 2>&1 & pid=$!
  local end=$((SECONDS+180))
  while ((SECONDS<end)); do
    kill -0 "$pid" 2>/dev/null || abort "$name server died"
    if curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok; then ready=1; break; fi
    sleep 3
  done
  ((ready)) || abort "$name health timeout"
  python3 - "$OUT" "$name" "$mode" <<'PY' || abort "$name request/timing/md5 gate failed"
import hashlib,json,pathlib,random,sys,time,urllib.request
out=pathlib.Path(sys.argv[1]); name,mode=sys.argv[2:]
def post(endpoint,body):
    req=urllib.request.Request('http://127.0.0.1:18080/'+endpoint,json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r: return json.load(r)
if mode=='std':
    prompt='Write a Python quicksort function and explain it.'
else:
    fixture=out/'prompt-2048.json'
    if not fixture.exists():
        # win2q gen(2048,2), normalized to an actual 2048-token prefix.
        r=random.Random(2); lines=[]; k=0
        while sum(len(l)+1 for l in lines)<2048*4:
            lines.append(f'uint64_t f{k}(uint64_t x) {{ return ((x ^ 0x{r.getrandbits(32):08x}ULL) + {k}) >> {k%13}; }}'); k+=1
        tokens=post('tokenize',{'content':'\n'.join(lines),'add_special':True})['tokens']
        assert len(tokens)>=2048, len(tokens)
        fixture.write_text(json.dumps(tokens[:2048]))
        print(f'{name} fixture generated_tokens={len(tokens)} retained_tokens=2048 seed=2',flush=True)
    prompt=json.loads(fixture.read_text())
for i in range(3 if mode=='std' else 2):
    start=time.monotonic()
    d=post('completion',{'prompt':prompt,'temperature':0,'seed':1,'n_predict':128 if mode=='std' else 32,'cache_prompt':False})
    (out/f'{name}.r{i}.json').write_text(json.dumps(d))
    t=d['timings']; md5=hashlib.md5(d['content'].encode()).hexdigest()
    if mode=='std':
        print(f"{name} r{i} decode={t['predicted_per_second']:.3f} tok/s md5={md5[:10]}",flush=True)
        assert md5[:10]=='9449ae5738', 'decode gold mismatch'
    else:
        pn=t['prompt_n']; cn=t['cache_n']; ms=t['prompt_ms']
        print(f'{name} r{i} prompt_n={pn} cache_n={cn} prompt_ms={ms:.3f} pp={pn*1000/ms:.3f} tok/s md5={md5} wall={time.monotonic()-start:.1f}s',flush=True)
        assert pn>=2047 and cn<=1 and ms>0, 'not a full 2048-token prefill'
PY
  stop_server
  sleep 3
}
leg O-std "$DST/build/bin/llama-server" std
leg O-2k-a "$DST/build/bin/llama-server" long
leg O-2k-b "$DST/build/bin/llama-server" long
leg S-2k "$SRC/build/bin/llama-server" long
python3 - "$OUT" "$patched" <<'PY'
import hashlib,json,pathlib,sys
out=pathlib.Path(sys.argv[1])
def rows(names): return [json.loads((out/f'{n}.r{i}.json').read_text()) for n in names for i in range(2)]
def hashes(ds): return [hashlib.md5(d['content'].encode()).hexdigest() for d in ds]
def pp(ds): return sum(d['timings']['prompt_n'] for d in ds)*1000/sum(d['timings']['prompt_ms'] for d in ds)
o=rows(['O-2k-a','O-2k-b']); s=rows(['S-2k'])
same=len(set(hashes(o)))==1
print('LAYOUT-INDEPENDENT='+('yes' if same else 'no'),flush=True)
print('S-2k VARIES='+('yes' if len(set(hashes(s)))>1 else 'no'),flush=True)
print(f'PP stock={pp(s):.3f} patched={pp(o):.3f} tok/s (total prompt tokens / total prompt seconds)',flush=True)
if not same or sys.argv[2]!='PASS':
    print('ABORT-WINDOW acceptance gate failed: patched order-check or model MD5s',flush=True)
    raise SystemExit(1)
PY
echo WIN2O-DONE
