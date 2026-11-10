#!/bin/bash
# Spark window mtp1 (patch 0068): is speculation with the MTP head invisible in the output?
#  S0 : plain decoding (no MTP), production env, traced: the reference bytes and logits rows
#  S1 : MTP head loaded, LLAMA_SPEC_EXACT=1 with one column (no drafts): does loading the head, the next-token
#       embeddings and the recurrent snapshots (n_rs_seq 3) move anything? Keep-tail prompt split on (default)
#  S2 : S1 with LLAMA_RS_KEEP_TAIL=0 (prompts chunk as without snapshots)
#  S3 : exact speculation on (4 columns: 3 drafts alone, 1 each at two clients, none at three or more), traced
#  S4 : S3 without the trace: speed        S5 : plain without the trace: speed
#  T01/T03/T31: tracecmp of S0 with S1 / S3 and of S3 with S1
# Usage: spark-mtp1.sh LEG...
cd /srv/glm-5.3-flash
exec 9>/srv/glm-5.3-flash/window.lock; flock -n 9 || { echo "ABORT: window.lock is held"; exit 1; }
export BD=${BD:-$HOME/nt/d121}
P="LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
T="LLAMA_SPEC_EXACT_TRACE=1"
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7"
T1="--temperature 1 --top-p 0.95"
SHORT=("specbench.py solo p0 p1 p2 p3 p4" "specbench.py solo p0 --n-predict 128" "specbench.py solo L16384"
       "specbench.py multi p0 p1 --rounds 0" "specbench.py multi p0 p1 p2 p3 p4 --rounds 0")
FULL=("${SHORT[@]}"
      "specbench.py solo p0 p2 p5 $T1 --seed 42" "specbench.py multi p0 p1 --rounds 1" "specbench.py multi p2 p5 --rounds 1 $T1 --seed 7"
      "specbench.py multi p0 p1 p2 --rounds 0" "specbench.py multi p0:60 p1:256 p2:150 --rounds 0" "specbench.py multi p3:40 p4:200 --rounds 0 $T1 --seed 11"
      "specbench.py follow p0 p6" "specbench.py follow p2 $T1 --seed 5" "specbench.py stop p0 p3" "specbench.py stop p2 p5 $T1 --seed 3"
      "specbench.py multi L16384 p0 p1 --rounds 0" "specbench.py solo p1 --n-probs 3 --n-predict 48")
SPEED=("specbench.py solo p0 p0 p1 p2 p3 p4" "specbench.py solo L16384" "specbench.py solo p0 p2 p5 $T1 --seed 42"
       "specbench.py multi p0 p1 --rounds 2" "specbench.py multi p0 p1 p2 --rounds 1" "specbench.py multi p0 p1 p2 p3 --rounds 1")
MS=${MTPSEED:-fl-mtp.k3sn}
for leg in "$@"; do
  case $leg in
    S0) bash mtp-leg.sh S0 5 1024000 "$P $T" "${FULL[@]}" ;;
    S1) SEED=$MS SAVE=mtp-seed.k3sn XARGS="$MTP" bash mtp-leg.sh S1 5 1024000 "$P $T LLAMA_SPEC_EXACT=1 LLAMA_SPEC_EXACT_COLS=1" "${SHORT[@]}" ;;
    S2) SEED=$MS XARGS="$MTP" bash mtp-leg.sh S2 5 1024000 "$P $T LLAMA_SPEC_EXACT=1 LLAMA_SPEC_EXACT_COLS=1 LLAMA_RS_KEEP_TAIL=0" "${SHORT[@]}" ;;
    S3) SEED=$MS XARGS="$MTP" bash mtp-leg.sh S3 5 1024000 "$P $T LLAMA_SPEC_EXACT=1 LLAMA_RS_KEEP_TAIL=0" "${FULL[@]}" ;;
    S4) SEED=$MS XARGS="$MTP" bash mtp-leg.sh S4 5 1024000 "$P LLAMA_SPEC_EXACT=1 LLAMA_RS_KEEP_TAIL=0" "${SPEED[@]}" ;;
    S5) bash mtp-leg.sh S5 5 1024000 "$P" "${SPEED[@]}" ;;
    T01) python3 -I tracecmp.py --subset legs-mtp/trace-S0.txt legs-mtp/trace-S1.txt | tail -8 ;;
    T02) python3 -I tracecmp.py --subset legs-mtp/trace-S0.txt legs-mtp/trace-S2.txt | tail -8 ;;
    T03) python3 -I tracecmp.py legs-mtp/trace-S0.txt legs-mtp/trace-S3.txt | tail -12 ;;
    *) echo "unknown leg $leg" ;;
  esac
done
echo "SPARK-MTP1-DONE $(date -u +%T)"
