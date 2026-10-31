#!/bin/bash
# win2c.sh (Spark): build the GLM-aware k3-pfdump, collect a 2048-token decode routing trace, check it, run place.py.
cd ~/llama-k3-bs && cp ~/win2/pfdump/* tools/k3-pfdump/ && export CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH && nice -n 19 cmake --build build -j 12 --target llama-k3-pfdump > ~/win2/buildpf.log 2>&1; echo "pfdump-build=$?"; ls build/bin | grep pfdump
M=/srv/glm-5.3/models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf; E=/srv/glm-5.3/store/glm53-q2-v1.k3xs; cp ~/win2/base.k3sn ~/win2/trace.k3sn
PF=$(ls build/bin/*pfdump | head -1)
env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$E LLAMA_EXPERT_STORE_SNAPSHOT=$HOME/win2/trace.k3sn LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 \
  LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 \
  LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 timeout 3000 $PF -m $M --ctx-size 65536 --batch-size 2048 --ubatch-size 512 \
  -ngl 999 --n-cpu-moe 999 --moe-cache-mib 80000 --threads 8 --threads-batch 20 --flash-attn on --fit off --load-mode mmap \
  -f ~/win2/glm-prompts.txt -n 2048 --ignore-eos --out ~/win2/decode.bin > ~/win2/trace.log 2>&1; echo "trace=$? $(ls -l ~/win2/decode.bin 2>/dev/null | awk '{print $5}') bytes"; tail -3 ~/win2/trace.log | cut -c1-160
$PF --check ~/win2/decode.bin 2>&1 | tail -6 | cut -c1-160
mkdir -p /srv/glm-5.3/placement && python3 ~/win2/place.py --trace ~/win2/decode.bin --store /srv/glm-5.3/store/glm53-q2-v2.k3xs --out /srv/glm-5.3/placement > ~/win2/place.log 2>&1; echo "place=$?"; tail -8 ~/win2/place.log | cut -c1-160; ls /srv/glm-5.3/placement
echo WIN2C-DONE
