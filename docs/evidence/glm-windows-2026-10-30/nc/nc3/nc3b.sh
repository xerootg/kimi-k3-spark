#!/bin/bash
# GH200 window nc3 (2026-11-10): patches 0061 (prompt ubatches in pieces), 0064 (a share for a prompt while its own
# class generates) and 0065 (prompts of one class take turns); A legs on the CI image (NEW_TAG = kimi-k3-spark 189b94b,
# patches 0060-0066), production config of #174 otherwise.
#  D1: pieces of 4 layers, share 0.12, lower classes' decode steps inside the share, same-class share 0.5, hold rule off; Juno's key at 10
#      (test keys generated here, never printed, deleted): the scenarios of nc2's HB/SLP legs.
#  HB3: the same scenarios on the production config (hold rule on, no pieces): the baseline.
#  TG1: D1's config with the decode graph reserved at full depth up front (LLAMA_PREFILL_SLICE_TG_CELLS=1048576) instead of
#      at 65536 cells growing on demand: its size, the peak GPU memory, the HBM tier, speeds. Decides TGC for the A legs.
#  D2: D1's config on the dev build with patch 0069 (out-b90: the lower classes' share split between their prompt and
#      their answers, accounts restart when the generating class changes, a solo step timed before the lower classes
#      join): ttft4 first on the fresh server, then D1's scenarios (without same3long and twolong), ttft4 and mixed
#      again after a same-class scenario.
#  A1, A2: acceptance (env TGC: LLAMA_PREFILL_SLICE_TG_CELLS for these legs, default = the patch's default), two server processes, no client classes (no keys in the test): per-client md5 equal to solo and
#      gold at N=1..5 with 16k/64k/100k mixes, two rounds; throughput against #174; a 100k decode (past the decode
#      graph's 64k-cell reserve).
set -u
G=/srv/glm-5.3-flash/nc; cd $G; mkdir -p keys; chmod 0700 keys
echo $$ > $G/nc3.pid   # started with setsid: this pid is the process group a deadman stops
NEW=ghcr.io/xerootg/llamacpp-spark:${NEW_TAG:-unset}
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
F="GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
SL="LLAMA_PREFILL_SLICE_LAYERS=${SLK:-4} LLAMA_PREFILL_SLICE_SHARE=${SLS:-0.12} LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=${SLM:-0.5} LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
DEV=$HOME/nc/out-a90   # dev build, same code as the patch series through 0066 (D1/HB3 run on it while the CI image builds)
CL="LLAMA_CLIENT_CLASSES=/keys/classes.txt LLAMA_SLOT_RESERVE=1"
K=$G/keys/classes.txt
PRIO=("prio_nc.py warm --classes $K --jn 512" "prio_nc.py junolong --classes $K --jn 512 --delay 1" "prio_nc.py prefill --classes $K --delay 10"
      "prio_nc.py junofirst --classes $K" "prio_nc.py junofirst --classes $K --jn 1024" "prio_nc.py ttft4 --classes $K" "prio_nc.py mixed --classes $K"
      "prio_nc.py samelong --classes $K --jn 1024 --delay 2" "prio_nc.py same3long --classes $K --jn 1024 --delay 2" "prio_nc.py shortbehind --classes $K --delay 5"
      "prio_nc.py twolong --classes $K --delay 5" "prio_nc.py nokey --classes $K")
FULL=("gold.py 2" "ekbench.py solo p0 p1 p2 p3 p4 L16384" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
      "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 L16384 --rounds 1"
      "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 2" "ekbench.py multi p0 p1 p2 p3 L65536 --rounds 1" "ekbench.py multi p1 p2 L16384 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536"
      "ekbench.py solo p0 p0 p0 p0" "ekbench.py solo L16384 L16384 L65536 L65536 --cache" "bench.py depth --lens 100000 --n-predict 64"
      "ekbench.py solo L100000" "ekbench.py multi p0 p1 p2 L100000 --rounds 1" "gold.py 1")
DEV2=$HOME/nc/out-b90
PRIO2=("prio_nc.py ttft4 --classes $K" "prio_nc.py warm --classes $K --jn 512" "prio_nc.py junolong --classes $K --jn 512 --delay 1" "prio_nc.py prefill --classes $K --delay 10"
      "prio_nc.py junofirst --classes $K" "prio_nc.py junofirst --classes $K --jn 1024" "prio_nc.py mixed --classes $K"
      "prio_nc.py samelong --classes $K --jn 1024 --delay 2" "prio_nc.py ttft4 --classes $K" "prio_nc.py mixed --classes $K" "prio_nc.py shortbehind --classes $K --delay 5")
mkkeys() { ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$K','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))" ); chmod 0644 $K; }
leakcheck() { python3 -c "
keys=[l.split()[0] for l in open('$K') if l.strip()]
t=open('$G/legs/server-$1.log',errors='replace').read()
print('server-$1.log', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"; }
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  case $leg in
    D1) mkkeys; OPT=$DEV CACHE_RAM=65536 bash nc-leg.sh D1 5 "$P $GR $F $CL $SL LLAMA_PREFILL_SLICE_LOG=1" "${PRIO[@]}"; leakcheck D1; rm -f $K ;;
    D2) mkkeys; OPT=$DEV2 CACHE_RAM=65536 bash nc-leg.sh D2 5 "$P $GR $F $CL $SL LLAMA_PREFILL_SLICE_LOG=1" "${PRIO2[@]}"; leakcheck D2; rm -f $K ;;
    HB3) mkkeys; OPT=$DEV CACHE_RAM=65536 bash nc-leg.sh HB3 5 "$P $GR $F $CL LLAMA_PRIORITY_HOLD_PROMPTS=1" "${PRIO[@]}"; leakcheck HB3; rm -f $K ;;
    TG1) OPT=$DEV CACHE_RAM=65536 bash nc-leg.sh TG1 5 "$P $GR $F $SL LLAMA_PREFILL_SLICE_TG_CELLS=1048576" "gold.py 2" "ekbench.py solo p0" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" \
           "ekbench.py solo L100000" "ekbench.py multi p0 p1 p2 L100000 --rounds 1" "gold.py 1" ;;
    A1) IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh A1s 5 "$P $GR $F $SL ${TGC:+LLAMA_PREFILL_SLICE_TG_CELLS=$TGC}" "${FULL[@]}" ;;
    A2) IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh A2s 5 "$P $GR $F $SL ${TGC:+LLAMA_PREFILL_SLICE_TG_CELLS=$TGC}" "${FULL[@]}" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $G/nc3.pid
echo "NC3B-DONE $(date -u +%T)"
