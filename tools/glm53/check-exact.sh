#!/bin/bash
# Compare g53x check's g/u/h/dn bytes against the shared CPU candidate, all Q2 slices.
# G53X_MUFU is the container path /w/mufu; MUFU_DIR is its host path.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$T/../.." && pwd)
W=${W:-$HOME/glm53-work}
MUFU_DIR=${MUFU_DIR:-$W/mufu}
CPU=$T/test-exact/g53ref_exact
mkdir -p "$T/test-exact"
g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math -I"$T/ref" -I"$T/../bitexact" "$T/g53ref_exact.cpp" -o "$CPU"
for spec in '3 17 17 18' '8 18 18 23' '75 17 17 23'; do
    read -r L TG TU TD <<< "$spec"
    P=$W/chk/g53x_UD-Q2_K_XL_L$L
    IN=$T/test-exact/sycl-input-L$L
    OUT=$T/test-exact/sycl-cpu-L$L
    mkdir -p "$IN"
    # Prepare a fresh input directory explicitly; compare helper does not overwrite computed stages.
    cp "$P.x" "$IN/x.bin"
    python3 - "$IN/ids.bin" <<'PY'
import struct,sys
from pathlib import Path
Path(sys.argv[1]).write_bytes(struct.pack('<8i',*range(8)))
PY
    "$CPU" slices "$MUFU_DIR" "$W/data/UD-Q2_K_XL_L$L" 8 "$TG" "$TU" "$TD" "$IN" "$OUT"
    python3 "$T/exact_test.py" sycl "$P" "$OUT"
done
