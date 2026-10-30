#!/bin/bash
# dsycl.sh CMD...: run CMD in intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04 with the iGPU, the work dir
# $W (default ~/glm53-work: llama/ = llama.cpp tree, data/ = extract.py output, build dirs) at /w, this dir at /w/tools
# and ../bitexact (expert_disk.h, read-only) at /bitexact. Unlimited locked memory (serveidx mlocks its disk-tier LRU);
# the K3X_* and G53X_* serving variables are passed through. DSYCL_ARGS: extra docker run arguments (e.g. a store mount
# "-v /nvme:/store:ro"). G53X_MUFU (g53x exact mode's MUFU tables, container path) defaults to /w/mufu ($W/mufu).
T=$(cd "$(dirname "$0")" && pwd); W=${W:-$HOME/glm53-work}; export G53X_MUFU=${G53X_MUFU:-/w/mufu}
exec docker run --rm -u $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --device /dev/dri -e HOME=/tmp \
  --ulimit memlock=-1:-1 $DSYCL_ARGS \
  -e IGC_ShaderDumpEnable -e IGC_DumpToCustomDir -e UR_L0_USE_IMMEDIATE_COMMANDLISTS -e G53X_PROF -e G53X_FUSEQ -e G53X_XIN -e G53X_HOSTW \
  -e G53X_EXACT -e G53X_MUFU \
  -e G53X_SYCL_LAUNCH -e G53X_WAIT -e G53X_NEO_TUNE -e G53X_ULLS -e G53X_SUBMIT -e G53X_MLOCKALL -e G53X_GPU_TIMEOUT_MS -e K3X_DISK_TIER -e K3X_DISK_STORE -e K3X_DISK_LRU_GB \
  -e K3X_DISK_DEADLINE_MS -e K3X_DISK_THREADS -e K3X_DISK_PIECE_KB -e K3X_NOSTREAM -e K3X_MAX_GB \
  -v $W:/w -v $T:/w/tools -v $T/../bitexact:/bitexact:ro -w /w intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04 \
  bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; $*"
