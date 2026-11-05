#!/bin/bash
# wait for the Q4 download, then repack the experts into a v1 store and verify 64 records
cd /srv/glm-5.3-flash
until grep -q -E "FETCH-OK|FETCH-SHA-FAIL" q4/fetch.log; do sleep 15; done
grep -q FETCH-OK q4/fetch.log || { echo PREP-FAIL fetch; exit 1; }
mkdir -p store snapshot
python3 -I /srv/glm-5.3/repack-experts.py --gguf q4/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --out store/glm53flash-q4.k3xs --verify 64 > repack.log 2>&1 && echo PREP-OK || echo PREP-FAIL repack
