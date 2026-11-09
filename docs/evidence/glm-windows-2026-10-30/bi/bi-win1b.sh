#!/bin/bash
# Window bi1, part b: layer outputs with fusion intact (filter l_out), production flags vs no 0050 flags.
cd /srv/glm-5.3-flash; L=legs-bi; mkdir -p $L
P="LLAMA_PREFILL_ISOLATE=1"
D="LLAMA_BI_DUMP_N=1234 LLAMA_BI_DUMP_STEPS=3"
R64=("solo p0 p1 p2 p3 --n-predict 64" "multi p0 p1 --rounds 0 --n-predict 64" "multi p0 p1 p2 --rounds 0 --n-predict 64" "multi p0 p1 p2 p3 --rounds 0 --n-predict 64")
RUNS=("solo p0 --n-predict 8" "multi p0 p1 --rounds 0 --n-predict 8" "multi p0 p1 p2 --rounds 0 --n-predict 8" "multi p0 p1 p2 p3 --rounds 0 --n-predict 8")
for leg in "$@"; do
  case $leg in
    F0) bash bi-leg.sh F0 4 4194304 "$P" "${R64[@]}" ;;
    L1) rm -f $L/dump-L1.txt; bash bi-leg.sh L1 4 4194304 "$P $D LLAMA_BI_DUMP_FILTER=l_out,result_output,hc_attn_post LLAMA_BI_DUMP=$PWD/$L/dump-L1.txt" "${RUNS[@]}" ;;
    N0) bash bi-leg.sh N0 4 4194304 "$P GGML_CUDA_Q8_REUSE=0 GGML_CUDA_FUSE_HC=0 GGML_CUDA_FUSE_GDN_STATE=0 GGML_CUDA_FUSE_KDA=0 GGML_CUDA_FUSE_CONV_STATE=0" "${R64[@]}" ;;
    N1) bash bi-leg.sh N1 4 4194304 "$P GGML_CUDA_DISABLE_FUSION=1" "${R64[@]}" ;;
  esac
done
echo "BI-WIN1B-DONE $(date -u +%T)"
