#!/bin/bash
# Window nd0: multi-stream baselines, production binary, remote tier off. A1 = parallel 1 standard (gold);
# A4 = parallel 4 plain; A4x = parallel 4 with the expert exact-columns knobs + N-column split.
set -u
OUT=/srv/glm-5.3-flash/legs-nd0
BIN=$HOME/llama-k3-el/build-el/bin/llama-server; export LD_LIBRARY_PATH="$HOME/llama-k3-el/build-el/bin"
source $HOME/nd/ndlib.sh
if start A1 1; then mb solo --tag A1 --n 1; stop; fi
if start A4 4; then mb solo --tag A4 --n 4; mb multi --tag A4 --n 2 --rounds 2; mb multi --tag A4 --n 4 --rounds 3; grep -a "split GEMV\|lookups" $OUT/server-A4.log | tail -2 | cut -c1-400; stop; fi
if start A4x 4 $EXACTN; then mb solo --tag A4x --n 4; mb multi --tag A4x --n 2 --rounds 2; mb multi --tag A4x --n 4 --rounds 3; grep -a "split GEMV\|lookups" $OUT/server-A4x.log | tail -2 | cut -c1-400; stop; fi
finish
echo WIN-ND0-DONE
