#!/usr/bin/env bash
# Run from the checkout ON SPARK, with the model/cache arguments of glm53-spark.yaml.
# Example: bash tools/glm53/placement.sh -m /path/to/model.gguf [production cache args]
set -euo pipefail
: "${PFDUMP:?Set PFDUMP to the patch-0005 llama-k3-pfdump binary}"
: "${PROMPT_FILE:?Set PROMPT_FILE to a representative prose/code prompt text file}"
root=$(cd "$(dirname "$0")/../.." && pwd)
output=${PLACEMENT_OUT:-/srv/glm-5.3/placement}
store=${GLM_STORE:-/srv/glm-5.3/store/glm53-q2-v2.k3xs}
mkdir -p "$output"
unset LLAMA_EXPERT_STORE_REMOTE LLAMA_EXPERT_STORE_REMOTE_SET LLAMA_EXPERT_STORE_TRACE
export LLAMA_EXPERT_STORE="$store"
export LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1
export LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3
export LLAMA_EXPERT_STORE_PREFETCH_MAX=5 LLAMA_EXPERT_STORE_SPLIT_LATE=8
"$PFDUMP" "$@" -f "$PROMPT_FILE" -n 2048 --out "$output/decode.bin"
python3 "$root/tools/glm53/place.py" --trace "$output/decode.bin" --store "$store" --out "$output" > "$output/planning.log"
bash "$output/deploy.sh"
# Source on the Spark before starting the engine, or merge generated engine.yaml into deployment.
printf 'Engine env: source %q\n' "$output/engine.env"
