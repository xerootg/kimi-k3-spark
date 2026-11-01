#!/bin/bash
# Link existing production libraries; never reconfigure/rebuild the production tree.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tree=${1:-$HOME/llama-k3-fd6}
out=${2:-$here/fattn-geometry}
export CUDACXX=${CUDACXX:-/usr/local/cuda/bin/nvcc}
[[ -x $CUDACXX ]] || { echo "ERROR: CUDA compiler unavailable: $CUDACXX" >&2; exit 1; }
[[ -f $tree/ggml/include/ggml.h ]] || { echo "ERROR: missing ggml public headers in $tree" >&2; exit 1; }
libdirs=()
while IFS= read -r dir; do libdirs+=("$dir"); done < <(find "$tree/build-fd4" -name 'libggml*.so*' -printf '%h\n' | sort -u)
((${#libdirs[@]})) || { echo 'ERROR: build-fd4 has no shared libggml libraries' >&2; exit 1; }
flags=()
for dir in "${libdirs[@]}"; do flags+=("-L$dir" "-Xlinker=-rpath" "-Xlinker=$dir"); done
"$CUDACXX" -std=c++17 -O2 -arch=sm_121 -I"$tree/ggml/include" \
    "$here/fattn-geometry.cu" "${flags[@]}" -lggml -lggml-cuda -lggml-base -o "$out"
echo "BUILD-FAGEOM OK $out"
