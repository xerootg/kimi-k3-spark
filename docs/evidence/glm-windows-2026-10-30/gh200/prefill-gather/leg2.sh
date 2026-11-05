#!/bin/bash
# usage: leg2.sh tag U hbm_mib max_batch bindir [extra env assignments...]
cd /srv/glm-5.3; out=pfub; tag=$1; U=$2; H=$3; MB=$4; BIN=$5; shift 5; EXTRA="$*"
BASE="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=$H LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0"
cp snapshot/glm53-q2-full.k3sn $out/snap-$tag.k3sn
(while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done) > $out/gpumem-$tag.txt & m=$!
echo "=== $tag U=$U hbm=$H $(date +%T)"
t0=$(date +%s)
env $BASE $EXTRA LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/$out/snap-$tag.k3sn LLAMA_EXPERT_STORE_MAX_BATCH=$MB $HOME/$BIN/build/bin/llama-server -m /srv/glm-5.3/q2/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --alias glm-5.3 --host 127.0.0.1 --port 18104 --ctx-size 524288 --parallel 1 --batch-size $U --ubatch-size $U --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 260000 --cache-ram 0 --threads 16 --threads-batch 32 --flash-attn on --jinja --no-warmup --fit off --load-mode mmap > $out/server-$tag.log 2>&1 &
pid=$!; ok=0
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:18104/health | grep -q ok && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 3; done
echo "LOAD $tag $(( $(date +%s) - t0 )) s"
if [ $ok = 1 ]; then python3 -I pf_client.py $tag || echo "CLIENT-FAIL $tag"; else echo "SERVER-FAIL $tag"; grep -a -i -E "error|abort|fail|out of memory" $out/server-$tag.log | tail -3 | cut -c1-200; fi
kill -INT $pid 2>/dev/null
for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done
kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null; kill $m
rm -f $out/snap-$tag.k3sn
echo "PEAK $tag $(sort -n $out/gpumem-$tag.txt | tail -1) MiB"
grep -a "compute buffer size" $out/server-$tag.log | cut -c1-200
echo LEG-DONE $tag
