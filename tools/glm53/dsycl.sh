#!/bin/bash
# dsycl.sh CMD...: run CMD in intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04 with the iGPU, the work dir
# $W (default ~/glm53-work: llama/ = llama.cpp tree, data/ = extract.py output, build dirs) at /w and this dir at /w/tools.
T=$(cd "$(dirname "$0")" && pwd); W=${W:-$HOME/glm53-work}
exec docker run --rm -u $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --device /dev/dri -e HOME=/tmp \
  -e IGC_ShaderDumpEnable -e IGC_DumpToCustomDir -e UR_L0_USE_IMMEDIATE_COMMANDLISTS -e G53X_PROF -e G53X_FUSEQ -e G53X_XIN \
  -v $W:/w -v $T:/w/tools -w /w intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04 \
  bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; $*"
