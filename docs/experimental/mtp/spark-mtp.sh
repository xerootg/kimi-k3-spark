#!/usr/bin/env bash
# Experimental launcher; run only after the exactness gates in RESULT.md.
set -euo pipefail
: "${S:?set S to the Spark llama-server binary}"
: "${M:?set M to the first model GGUF shard}"
: "${E:?set E to the expert store}"
mtp_args=(--spec-type none)
if [[ ${GLM53_MTP_EXPERIMENT:-0} == 1 ]]; then
    mtp_args=(--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-min 1 --spec-draft-p-min 0 --spec-draft-sampling greedy)
fi
exec env LLAMA_EXPERT_STORE="$E" "$S" -m "$M" --parallel 1 --ctx-size 32768 \
    --batch-size 2048 --ubatch-size 512 --n-gpu-layers 999 --n-cpu-moe 999 \
    --moe-cache-mib 76000 --flash-attn on --fit off --temp 0 "${mtp_args[@]}" "$@"
