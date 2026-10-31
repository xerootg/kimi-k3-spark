#!/bin/bash
# win2s.sh (Spark): patch 0019 (SIGTERM = SIGINT, LLAMA_EXPERT_STORE_FAST_EXIT): time from SIGTERM to exit, native server.
cd ~/llama-k3-bs && git apply --check ~/win2/0019-sigterm-fast-exit.patch && git apply ~/win2/0019-sigterm-fast-exit.patch && export CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH && nice -n 19 cmake --build build -j 12 --target llama-server > ~/win2/build19.log 2>&1; echo "build19=$?"; [ "$(tail -c 200 ~/win2/build19.log | grep -c 'Built target\|Linking')" -ge 1 ] || { echo "ABORT-WINDOW: patch or build failed"; exit 1; }
cd /srv/glm-5.3; M=models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf
for fe in 0 1 0 1; do tag=FE$fe$RANDOM; cp legs/base.k3sn legs/$tag.k3sn
  env LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/legs/$tag.k3sn LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=1 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_FAST_EXIT=$fe ~/llama-k3-bs/build/bin/llama-server -m $M --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size 2048 -ngl 999 --n-cpu-moe 999 --moe-cache-mib 80000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 > legs/$tag.log 2>&1 & pid=$!
  until curl -s -m 2 127.0.0.1:18080/health | grep -q ok; do sleep 3; kill -0 $pid 2>/dev/null || { echo "$tag died"; break; }; done
  curl -s -m 600 127.0.0.1:18080/completion -H 'Content-Type: application/json' -d '{"prompt":"hi","n_predict":8,"temperature":0}' > /dev/null
  t0=$(date +%s.%N); kill -TERM $pid; wait $pid 2>/dev/null; t1=$(date +%s.%N); echo "FAST_EXIT=$fe: SIGTERM to exit $(python3 -c "print(round($t1-$t0,1))") s; snapshot $(stat -c %y legs/$tag.k3sn | cut -c12-19) $(grep -a -c 'snapshot (exit)' legs/$tag.log) exit-line"; sleep 5
done
echo WIN2S-DONE
