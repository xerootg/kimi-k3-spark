#!/bin/bash
# spark-nc14.sh: third runner of nc-s8, for point 69d = p0069 plus only the CUDA hunks of 0070 built IN THE ORIGINAL tree
# ~/bis/src (bisbuild3.sh), so the objects of every untouched file are byte-identical to p0069 and only the CUDA
# backend (and libllama-common, which 0070 does not touch here) can differ. 69c (second runner) was the same source in a
# copied tree, ~/bis/src2, whose path is baked into every object: every library differed, so it tells nothing alone.
# spark-nc13.sh: second runner of Spark window nc-s8, started after spark-nc12.sh finished, for the extra point 69c =
# p0069 plus only the two CUDA hunks of 0070 (common.cuh member, owner check), built in ~/bis/src2 (bisbuild2.sh), to
# tell a recompiled CUDA backend from the rest of 0070. Otherwise identical to spark-nc12.sh, whose header follows.
# Spark window nc-s8 (2026-11-14): bisecting the one-client cost of the gfbb8831 build. Window nc-s7 measured one
# client with short prompts 0.5-0.9 % slower on the binaries of image gfbb8831 (patches through 0076) than on
# g08d176f (through 0067), with LLAMA_KPOOL_OWN_SEQS=1 and without; the control's spread from process to process was
# 0.0-0.4 %. Here every stack point between the two is a native sm_121 build on this box from one tree (~/bis, see
# bisbuild.sh): p0067 (the control), p0069, p0070, p0071, p0072, p0073, p0074, p0075, p0076. Every leg is a fresh server
# with today's production args and env (five slots of 200k, --ctx-size 1024000, column split from three clients,
# batch-invariant decode, prefill isolate, elastic KV with refuse, the class file with one reserved slot, the hold
# rule, LLAMA_KPOOL_OWN_SEQS=1, which builds before 0071 ignore), running the five short prompts alone twice in a row
# (pass one, then the warm pass two) as a priority-1 client (class claude-code). Legs are named <point>.<round>; a
# round is the nine points once, in a different order per round so a drift of the box cannot line up with a point.
# <point>.m is the two-client aggregate (p0 p1, two rounds). c08 and cfb are the binaries of the CI images g08d176f
# and gfbb8831 (~/nc/ci-*), to tie the native builds to the images nc-s7 measured.
# Env DEADLINE (epoch seconds): no leg and no client command starts after it. Test keys are generated here, never
# printed, deleted after each leg. glm53-spark at 0, bare servers on 127.0.0.1:18080 under window.lock.
set -u
cd /srv/glm-5.3-flash
P="LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3 LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1 LLAMA_PRIORITY_HOLD_PROMPTS=1 LLAMA_KPOOL_OWN_SEQS=1"
OUT=legs-nc3; mkdir -p $OUT; echo $$ > $OUT/runner.pid
exec 8>/srv/glm-5.3-flash/window.lock; flock -n 8 || { echo "window.lock is held by someone else, not starting"; exit 1; }
export WINDOW_LOCK_HELD=1 # held here for the whole window; nc3-leg.sh skips its own flock
echo "nc-s8 runner $$ holds window.lock from $(date -u +%T), legs: $*"
KD=$PWD/keys-nc; K=$KD/classes.txt; KF=$KD/key-cc.txt
CL="LLAMA_CLIENT_CLASSES=$K LLAMA_SLOT_RESERVE=1"
S=("ekbench3.py solo p0 p1 p2 p3 p4 --key-file $KF" "ekbench3.py solo p0 p1 p2 p3 p4 --key-file $KF")
M=("ekbench3.py multi p0 p1 --rounds 2 --key-file $KF")
mkkeys() { mkdir -p $KD; chmod 0700 $KD; ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
keys=[(secrets.token_urlsafe(24),p,n) for n,p in rows]
open('$K','w').write(''.join('%s %d %s\n'%k for k in keys))
open('$KF','w').write([k[0] for k in keys if k[2]=='claude-code'][0]+'\n')" ); }
leakcheck() { python3 -c "
keys=[l.split()[0] for l in open('$K') if l.strip()]
t=open('$OUT/server-$1.log',errors='replace').read()
print('server-$1.log', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"; }
bd() { case $1 in 69c) echo $HOME/bis/p0069c;; 69d) echo $HOME/bis/p0069d;; 67|69|70|71|72|73|74|75|76) echo $HOME/bis/p00$1;; c08) echo $HOME/nc/ci-08d176f;; cfb) echo $HOME/nc/ci-fbb8831;; *) echo /nonexistent;; esac; }
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  pt=${leg%%.*}; rd=${leg##*.}; BDIR=$(bd $pt)
  [ -x $BDIR/bin/llama-server ] || { echo "no binary for leg $leg ($BDIR)"; continue; }
  mkkeys
  if [ "$rd" = m ]; then
    BD=$BDIR bash nc3-leg.sh S$leg 5 1024000 "$P $CL" "${M[@]}"
  else
    BD=$BDIR bash nc3-leg.sh S$leg 5 1024000 "$P $CL" "${S[@]}"
  fi
  leakcheck S$leg; rm -f $K $KF
done
rm -f $K $KF $OUT/runner.pid
echo "SPARK-NC14-DONE $(date -u +%T)"
