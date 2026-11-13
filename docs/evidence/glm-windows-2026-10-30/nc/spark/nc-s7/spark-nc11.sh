#!/bin/bash
# Spark window nc-s7, second runner (2026-11-11): the first runner (spark-nc10.sh) showed one client on a fresh server
# 0.3-0.8 % slower with A' (two processes, 15.36-15.97 tok/s) than the single control process (15.43-16.05). One
# control process cannot say whether that is the code or the process. Short legs, fresh server each, the five short
# prompts alone twice in a row:
#  QC1, QC2: the control again (binaries of g08d176f, today's flags): its spread from process to process
#  TA2:      A' again (binaries of gfbb8831, today's flags + LLAMA_KPOOL_OWN_SEQS=1)
#  TN1:      the binaries of gfbb8831 with today's flags and WITHOUT the variable: build or switch
# Env DEADLINE (epoch seconds): no leg and no client command starts after it.
set -u
cd /srv/glm-5.3-flash; NEW=$HOME/nc/ci-fbb8831; OLD=$HOME/nc/ci-08d176f
P="LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3 LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
H="LLAMA_PRIORITY_HOLD_PROMPTS=1"; KO="LLAMA_KPOOL_OWN_SEQS=1"
OUT=legs-nc2; mkdir -p $OUT; echo $$ > $OUT/runner.pid
S=("ekbench.py solo p0 p1 p2 p3 p4" "ekbench.py solo p0 p1 p2 p3 p4")
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  case $leg in
    QC1|QC2) BD=$OLD bash nc2-leg.sh $leg 5 1024000 "$P $H" "${S[@]}" ;;
    TA2) BD=$NEW bash nc2-leg.sh TA2 5 1024000 "$P $H $KO" "${S[@]}" ;;
    TN1) BD=$NEW bash nc2-leg.sh TN1 5 1024000 "$P $H" "${S[@]}" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $OUT/runner.pid
echo "SPARK-NC11-DONE $(date -u +%T)"
