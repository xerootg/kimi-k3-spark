#!/bin/bash
# GH200 window nc3, debugging legs (2026-11-10): Juno's bytes differed from solo in "mixed" on the 0069 draft (out-b90).
# One leg = D2's config with one thing changed; prio_nc_d.py keeps the generated text so the first differing token is known.
#  X1: D2's config.  XF: without the fusion/Q8-reuse flags.  XT: decode graph reserved at full depth.
#  XS: without GGML_CUDA_KV_STAGE.  XO: old build (out-a90, patch 0066) with D2's scenarios.  XN: same-class share off.
# env SCEN: "16k" (mixed with a 16k prompt, three times), "64k" (mixed as in D2, twice), "both"
set -u
G=/srv/glm-5.3-flash/nc; cd $G; mkdir -p keys; chmod 0700 keys
echo $$ > $G/nc3.pid
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
F="GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
SL="LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_SHARE=0.12 LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
SLN="LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_SHARE=0.12 LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
CL="LLAMA_CLIENT_CLASSES=/keys/classes.txt LLAMA_SLOT_RESERVE=1"
K=$G/keys/classes.txt
DEV=$HOME/nc/out-a90; DEV2=$HOME/nc/out-b90
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
  D="--dump $G/legs/dump-$leg"
  S16=("prio_nc_d.py mixed --classes $K --long L16384 $D" "prio_nc_d.py mixed --classes $K --long L16384 $D" "prio_nc_d.py mixed --classes $K --long L16384 $D")
  S64=("prio_nc_d.py mixed --classes $K $D" "prio_nc_d.py mixed --classes $K $D")
  case ${SCEN:-16k} in 16k) SC=("${S16[@]}") ;; 64k) SC=("${S64[@]}") ;; *) SC=("${S16[@]}" "${S64[@]}") ;; esac
  SC=("prio_nc_d.py warm --classes $K $D" "${SC[@]}")
  mkkeys
  case $leg in
    X1*) OPT=$DEV2 CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $F $CL $SL LLAMA_PREFILL_SLICE_LOG=1" "${SC[@]}" ;;
    XF*) OPT=$DEV2 CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $CL $SL LLAMA_PREFILL_SLICE_LOG=1" "${SC[@]}" ;;
    XT*) OPT=$DEV2 CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $F $CL $SL LLAMA_PREFILL_SLICE_LOG=1 LLAMA_PREFILL_SLICE_TG_CELLS=1048576" "${SC[@]}" ;;
    XS*) OPT=$DEV2 CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 $F $CL $SL LLAMA_PREFILL_SLICE_LOG=1" "${SC[@]}" ;;
    XO*) OPT=$DEV CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $F $CL $SL LLAMA_PREFILL_SLICE_LOG=1" "${SC[@]}" ;;
    XN*) OPT=$DEV2 CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $F $CL $SLN LLAMA_PREFILL_SLICE_LOG=1" "${SC[@]}" ;;
    XE*) OPT=${OPTDIR:-$DEV2} CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $F $CL $SL LLAMA_PREFILL_SLICE_LOG=1 ${EXTRAENV:-}" "${SC[@]}" ;;   # env OPTDIR, EXTRAENV
    *) echo "unknown leg $leg" ;;
  esac
  leakcheck $leg; rm -f $K
done
rm -f $K $G/nc3.pid
echo "NC3C-DONE $(date -u +%T)"
