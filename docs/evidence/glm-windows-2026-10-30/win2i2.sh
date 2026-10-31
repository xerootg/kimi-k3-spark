#!/bin/bash
# Spark window i2: copied native candidate, frozen cache, one server per leg.
set -euo pipefail
abort() { echo "ABORT-WINDOW $*"; exit 1; }
trap 'abort "line $LINENO failed"' ERR
cd /srv/glm-5.3
START=$SECONDS
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
for cmd in python3 git cmake timeout curl cp env; do
  command -v "$cmd" >/dev/null || abort "missing command $cmd"
done
budget() { local left=$((2250 - SECONDS + START)); ((left > 0)) || abort 'runtime budget exhausted'; echo "$left"; }
echo 'PRECONDITION placement: c2 routing trace may still be queued; checking report source'
echo 'place.py writes owners.csv, spark.set, nuc{2,3}-{ram,ssd}.set, report.json, engine.env/yaml, deploy.sh; with --store also nuc{2,3}-ram.idx.'
echo 'NUC g53x needs nuc2-ram.set; g53cs needs nuc2-ram.idx from its identical v2 store. Both need store/sidecar and verified MUFU tables; startup emits served.set.'
echo 'Spark uses owner-filtered RAM-only nuc2 sets intersected with actual nuc2.served.set, never engine.env or all-record advertisements.'
for f in owners.csv spark.set nuc2-ram.set nuc2-ssd.set nuc3-ram.set nuc3-ssd.set report.json nuc2.served.set; do
  [[ -f placement/$f ]] || abort "missing placement/$f (run c2 and fetch NUC served.set)"
done
mkdir -p legs
[[ ! -e placement/nuc2.ram-served.set ]] || abort 'existing placement/nuc2.ram-served.set; archive prior window outputs first'
python3 - <<'PY' || abort 'placement invalid'
import json,csv,pathlib
p=pathlib.Path('placement')
source=json.loads((p/'report.json').read_text())['source']
assert source == 'trace' or source.startswith('synthetic'), source
print('PRECONDITION placement source:',source, '(SYNTHETIC: no trace performance claim)' if source.startswith('synthetic') else '(routing trace)')
def pairs(name):
    rows=[tuple(map(int,s.split())) for s in (p/name).read_text().splitlines() if s.strip()]
    assert all(len(r)==2 and 3<=r[0]<78 and 0<=r[1]<256 for r in rows), name
    assert len(rows)==len(set(rows)), name
    return set(rows)
sets={n:pairs(n+'.set') for n in ['spark','nuc2-ram','nuc2-ssd','nuc3-ram','nuc3-ssd']}
allpairs=set().union(*sets.values())
assert len(allpairs)==19200 and sum(map(len,sets.values()))==19200
owners=list(csv.DictReader((p/'owners.csv').open()))
assert len(owners)==19200
assert {(int(r['layer']),int(r['expert'])) for r in owners}==allpairs
assert all((int(r['layer']),int(r['expert'])) in sets[r['owner']] for r in owners)
served=pairs('nuc2.served.set')
remote=sets['nuc2-ram'] & served
assert remote, 'no supported nuc2-owned rows'
# Only a derived engine input; stores and original placement files stay untouched.
(p/'nuc2.ram-served.set').write_text(''.join(f'{l} {e}\n' for l,e in sorted(remote)))
print('PRECONDITION RAM-only supported owner rows:',len(remote))
PY
python3 - <<'PYPROC' || abort 'existing llama-server -m process or occupied port 18080'
import pathlib,socket
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try: args=p.read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError): continue
    assert not (args and b'llama-server' in args[0] and b'-m' in args), str(p)
with socket.socket() as s:
    s.bind(('127.0.0.1',18080))
print('PRECONDITION no llama-server -m process; port 18080 free')
PYPROC
echo 'PRECONDITION baseline ~/llama-k3-c: expected 0001-0006 + 0009 + 0012 + 0011 v2 + 0012b + 0012c (uncommitted stack retained by cp -a)'
SRC=$HOME/llama-k3-c
DST=$HOME/llama-k3-r2
PATCH=$HOME/win2/0013-remote-tier-glm.patch
HEDGE=$HOME/win2/0013b-remote-hedge.patch
[[ -s $HEDGE ]] || abort 'missing ~/win2/0013b-remote-hedge.patch'
[[ -d $SRC/.git && -s $SRC/build/CMakeCache.txt && -x $SRC/build/bin/llama-server ]] || abort 'baseline native tree/build missing'
[[ -s $PATCH ]] || abort 'missing fixed ~/win2/0013-remote-tier-glm.patch'
for marker in remote_set_key remote_late 0x7f800000U; do
  grep -q "$marker" "$PATCH" || abort "0013 lacks fixed marker $marker"
done
for marker in LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO LLAMA_EXPERT_STORE_HOOK_LEAN; do
  grep -q "$marker" "$SRC/src/llama-moe-cache.cpp" || abort "baseline lacks $marker"
done
[[ ! -e $DST ]] || abort 'candidate ~/llama-k3-r2 already exists; archive before rerunning'
git -C "$SRC" apply --check "$PATCH" || abort '0013 not applicable to baseline'
echo 'PRECONDITION baseline compatibility OK; patch lineage must be established by earlier windows, not git clean (stack is uncommitted).'
[[ ${NUC_EXACT_VERIFIED:-0} == 1 ]] || abort 'laptop must attest fresh NUC G53X_EXACT=1 or verified =2, correct store and verified GB10 MUFU tables (NUC_EXACT_VERIFIED=1)'
echo 'PRECONDITION exact attestation supplied; NaN health probe proves protocol only; every H1 response still must pass md5 9449ae5738'
echo 'PRECONDITION NUC: laptop must scale glm53-nuc2 to 1 first; Spark never invokes kubectl.'
python3 - <<'PY' || abort 'nuc2:19600 k3x probe did not answer within 120 s'
import socket,struct,time,math
end=time.monotonic()+120
# No HTTP health endpoint or handshake exists. A k=1 invalid gid must return
# one 6144-float NaN sentinel row: protocol health only, not exactness proof.
while time.monotonic()<end:
    try:
        with socket.create_connection(('192.168.4.249',19600),timeout=min(2,end-time.monotonic())) as s:
            s.settimeout(min(2,max(.001,end-time.monotonic())))
            s.sendall(struct.pack('<17i',1,-1,*([0]*15))+bytes(6144*4))
            data=bytearray()
            while len(data)<6144*4:
                s.settimeout(min(2,max(.001,end-time.monotonic())))
                b=s.recv(6144*4-len(data))
                if not b: raise OSError('short reply')
                data.extend(b)
            assert all(math.isnan(v[0]) for v in struct.iter_unpack('<f',data))
            print('PRECONDITION nuc2:19600 (192.168.4.249) k3x NaN sentinel health OK')
            break
    except (OSError,AssertionError):
        time.sleep(min(1,max(0,end-time.monotonic())))
else:
    raise SystemExit(1)
PY
[[ -s snapshot/glm53-q2.k3sn ]] || abort 'missing frozen snapshot'
[[ -r store/glm53-q2-v1.k3xs && -r models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf ]] || abort 'missing local store or model'
echo 'PRECONDITION frozen snapshot, local store and model OK; stores are read only inputs'
# Do not clone: the baseline patches are uncommitted. Incremental build reuses
# copied objects; reconfigure first because CMakeCache embeds absolute paths.
cp -a "$SRC" "$DST"
git -C "$DST" apply "$PATCH" || abort '0013 apply failed on copy'
git -C "$DST" apply --check "$HEDGE" || abort '0013b not applicable after 0013'
git -C "$DST" apply "$HEDGE" || abort '0013b apply failed on copy'
export CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH
# Rewrite copied CMake path references, without changing baseline or stores.
python3 - "$SRC" "$DST" <<'PY'
import pathlib,sys
src,dst=sys.argv[1:]
b=pathlib.Path(dst)/'build'
for p in b.rglob('*'):
    if p.is_file() and not p.is_symlink() and (p.name in ('CMakeCache.txt','Makefile','build.ninja','rules.ninja') or p.suffix in ('.cmake','.make')):
        data=p.read_bytes()
        if src.encode() in data: p.write_bytes(data.replace(src.encode(),dst.encode()))
PY
left=$(budget)
timeout "${left}s" bash -c 'cd "$1" && cmake -S . -B build && nice -n 19 cmake --build build -j 12 --target llama-server' _ "$DST" > "$HOME/win2/build13b.log" 2>&1 || abort '0013+0013b configure/build failed (~/win2/build13b.log)'
echo 'build13b=0'
BIN=$DST/build/bin/llama-server
[[ -x $BIN ]] || abort 'candidate binary missing'
cp snapshot/glm53-q2.k3sn legs/I2-base.k3sn
P='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":128,"cache_prompt":false}'
# Remove inherited remote knobs so off/shadow/late-post cannot contaminate legs.
for v in ${!LLAMA_EXPERT_STORE_REMOTE@}; do unset "$v"; done
STORE=glm53-q2-v1.k3xs
SZ=1 HL=1
leg() { local name=$1 deadline=$2 i left ready=0
  left=$(budget)
  cp legs/I2-base.k3sn "legs/$name.k3sn"
  local -a remote=()
  if ((deadline)); then
    # Engine parser requires literal IPv4, not a DNS hostname.
    remote=(LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19600
      LLAMA_EXPERT_STORE_REMOTE_SET=/srv/glm-5.3/placement/nuc2.ram-served.set
      LLAMA_EXPERT_STORE_REMOTE_HEDGE=1 LLAMA_EXPERT_STORE_REMOTE_EXACT=1
      LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US="$deadline" LLAMA_EXPERT_STORE_REMOTE_ROWS=16)
  fi
  echo "$name remote deadline_us=$deadline SZ=$SZ HL=$HL"
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/$STORE LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/legs/$name.k3sn \
  LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 \
  LLAMA_EXPERT_STORE_RESERVE_MIB=4096 ${PROF:+GGML_CUDA_OP_PROFILE=$PROF} LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=$SZ LLAMA_DSA_SHORT_CTX_SKIP=0 LLAMA_EXPERT_STORE_HOOK_LEAN=$HL LLAMA_EXPERT_STORE_HOOK_STATS=1 "${remote[@]}" "$BIN" -m models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
  --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size 512 -ngl 999 --n-cpu-moe 999 \
  --moe-cache-mib 80000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 \
  > "legs/$name.log" 2>&1 & pid=$!
  local until_time=$((SECONDS+180))
  while ((SECONDS<until_time)); do
    left=$(budget)
    kill -0 "$pid" 2>/dev/null || abort "$name server died"
    if curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok; then ready=1; break; fi
    sleep 3
  done
  ((ready)) || abort "$name health timeout"
  for i in 0 1 2; do
    left=$(budget); ((left>180)) && left=180
    curl -fsS -m "$left" 127.0.0.1:18080/completion -H 'Content-Type: application/json' -d "$P" > "legs/$name.r$i.json" || abort "$name request $i failed"
  done
  python3 - "$name" <<'PY' || abort "$name response/md5 gate failed"
import json,hashlib,sys
name=sys.argv[1]
for i in range(3):
    d=json.load(open(f'legs/{name}.r{i}.json'))
    md5=hashlib.md5(d['content'].encode()).hexdigest()[:10]
    print(name,f'r{i}',round(d['timings']['predicted_per_second'],3),'tok/s md5',md5,flush=True)
    assert md5=='9449ae5738', 'gold mismatch'
PY
  stop_server
  # Shutdown flushes final cumulative statistics. Print box counters, then
  # aggregate latest per-layer values (never sum repeated cumulative dumps).
  grep -a -E 'remote box .*posts,' "legs/$name.log" | tail -1 || :
  grep -a -E 'remote experts.*layers with candidates' "legs/$name.log" | tail -1 || :
  grep -a 'hit rate' "legs/$name.log" | tail -1 || :
  grep -a -E 'split GEMV|hedge total|hedge breaker|hedge drain|host bank real (readiness|bucket|histogram)|expert store fill dispatch' "legs/$name.log" || :
  python3 - "$name" <<'PY'
import re,sys
name=sys.argv[1]; layers={}
for line in open(f'legs/{name}.log',errors='replace'):
    m=re.search(r'hedge total layer=(\d+) posts=(\d+) hedged=(\d+) remote_wins=(\d+) local_wins=(\d+) late_remote=(\d+) NaN=(\d+)',line)
    if m: layers[int(m[1])]=tuple(map(int,m.groups()[1:]))
a=list(map(sum,zip(*layers.values()))) if layers else [0]*6
print(name,'hedge counters posts/hedged/remote_wins/local_wins/late_remote/NaN:',*a)
if name.startswith('H1'):
    assert a[0]>0 and a[2]>0, 'hedge GATE-FAIL: require posts and remote wins, hash alone is insufficient'
PY
  echo "$name MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
  sleep 5
}
leg H0 0
leg H1 2500
leg H0b 0
leg H1b 2500
leg H1c 4000
echo WIN2I2-DONE
