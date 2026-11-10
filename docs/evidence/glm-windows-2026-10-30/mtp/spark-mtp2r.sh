#!/bin/bash
# Spark window mtp2, reproduction legs for the abort of leg S3 ("expert store: layer 24 expert 187 has slot -1 after the
# plan (n_slots 8682, 32 ids, promote 1)", at the start of "multi L16384 p0 p1"). Build f121 = e121 plus the diagnostic
# LLAMA_EXPERT_STORE_PLAN_CHECK=1 (ids copied before the plan and compared at the failure, a re-entrancy counter, the
# aborting thread's backtrace).
#  R3 : S3's server (exact speculation, 4 columns, traced), S3's last request before the abort, then the 16k mix three times
#  R1 : the same with one column (MTP loaded, no drafts)
#  R0 : the same on a plain server (no MTP)
# Usage: spark-mtp2r.sh LEG...
cd /srv/glm-5.3-flash
exec 9>/srv/glm-5.3-flash/window.lock; flock -n 9 || { echo "ABORT: window.lock is held"; exit 1; }
export BD=${BD:-$HOME/nt/f121}
P="LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
T="LLAMA_SPEC_EXACT_TRACE=1 LLAMA_EXPERT_STORE_PLAN_CHECK=1"
X="LLAMA_SPEC_EXACT=1 LLAMA_RS_KEEP_TAIL=0"
mtp() { echo "--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min $1"; }
T1="--temperature 1 --top-p 0.95"
REPRO=("specbench.py stop p2 p5 $T1 --seed 3" "specbench.py multi L16384 p0 p1 --rounds 0" "specbench.py multi L16384 p0 p1 --rounds 0"
       "specbench.py multi L16384 p0 p1 --rounds 0")
MS=${MTPSEED:-fl-mtp.k3sn}
leg() {
  case $1 in
    R3) SEED=$MS XARGS="$(mtp 0.7)" bash mtp-leg.sh R3 5 1024000 "$P $T $X" "${REPRO[@]}" ;;
    R1) SEED=$MS XARGS="$(mtp 0.7)" bash mtp-leg.sh R1 5 1024000 "$P $T LLAMA_SPEC_EXACT=1 LLAMA_SPEC_EXACT_COLS=1 LLAMA_RS_KEEP_TAIL=0" "${REPRO[@]}" ;;
    R0) bash mtp-leg.sh R0 5 1024000 "$P $T" "${REPRO[@]}" ;;
    *) echo "unknown leg $1" ;;
  esac
}
for l in "$@"; do leg $l; done
echo "SPARK-MTP2R-DONE $(date -u +%T)"
