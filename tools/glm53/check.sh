#!/bin/bash
# check.sh: every slice through ggml CPU, ggml SYCL, g53x and g53c (ids 0..7, same x), then glmref: fp64 reference,
# each implementation's arithmetic against the exact sum over its own q8_1 inputs, and the implementations against each other.
set -euo pipefail
T0=$(cd "$(dirname "$0")" && pwd); W=${W:-$HOME/glm53-work}; cd $W; mkdir -p chk
SL=("UD-Q2_K_XL_L3 17 17 18" "UD-Q2_K_XL_L8 18 18 23" "UD-Q2_K_XL_L75 17 17 23" "UD-Q4_K_XL_L3 12 12 13" "UD-Q4_K_XL_L8 13 13 14" "UD-Q4_K_XL_L75 12 12 14")
cmd=""
for s in "${SL[@]}"; do set -- $s
  ex=${G53X_EXACT:-2}; [[ "$1" == UD-Q4* ]] && ex=0
  cmd+="build-sycl/glmbench SYCL0 data/$1 $2 $3 $4 8 1 1 --check chk/sycl_$1 2>/dev/null; G53X_EXACT=$ex ./g53x data/$1 $2 $3 $4 check chk/g53x_$1 2>&1 | grep fused; "
done
$T0/dsycl.sh "set -euo pipefail; $cmd"
for s in "${SL[@]}"; do set -- $s
  build-cpu/glmbench CPU data/$1 $2 $3 $4 8 1 8 --check chk/cpu_$1 2>/dev/null
  c=""; [ "$4" = 14 ] || { ./g53c data/$1 $2 $3 $4 8 check chk/g53c_$1 2>/dev/null; c=chk/g53c_$1; }
  echo "== $1"; build-cpu/glmref data/$1 $2 $3 $4 chk/g53x_$1 chk/sycl_$1 chk/cpu_$1 $c | grep -v stage
done

if [[ ${G53X_EXACT:-2} != 0 ]]; then
  "$T0/check-exact.sh"
fi
