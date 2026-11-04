# Task: GH200 prefill levers 1 and 2 (GLM-5.3 Q2, llama.cpp fork), measure and report

You work alone, unattended, with a hard deadline: **write RESULT.md in this directory within 38 minutes of starting**,
even if some legs did not run. Numbers over prose. Do not ask questions; decide and note the decision.

## Machine
- `ssh ubuntu@192.222.50.246` (gh200-dev): aarch64 Grace + 1 H100-class GPU (96 GB HBM3, sm_90), 525 GB host RAM,
  64 KiB pages. Passwordless sudo, but you should not need it.
- The production server (Kubernetes deployment glm53-gh200) is scaled to 0 by the supervisor. The GPU is free.
  **Never touch Kubernetes (no kubectl).** The supervisor restores production afterwards.
- Working dir on the box: `/srv/glm-5.3` (model `q2/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf`, expert store
  `store/glm53-q2-v1.k3xs`, snapshot `snapshot/glm53-q2-full.k3sn`).
- Built server with the production patch stack: `~/llama-pf/build/bin/llama-server` (source in `~/llama-pf`).

## Hard rules
1. **Never modify or delete anything in `/srv/glm-5.3/snapshot/`, `/srv/glm-5.3/store/`, `/srv/glm-5.3/q2/`.** Each
   server start uses its own copy of the snapshot: `cp snapshot/glm53-q2-full.k3sn <yourdir>/snap-<tag>.k3sn`.
2. Put all your files under `/srv/glm-5.3/pfub/` on the box and in this local directory.
3. Run servers detached on the box (`setsid nohup ... &`) and log to files. Stop them with `kill -INT <pid>` (explicit
   pid), then `kill -KILL <pid>` after 60 s. **Never use `pkill -f` or `pgrep -f` with a pattern that also matches your
   own ssh command line**; use `pgrep -f "[l]lama-server"` (bracket trick) or saved pids.
4. One server at a time. Before starting one, check `pgrep -af "[l]lama-server"` is empty and `nvidia-smi` shows
   under 1 GiB used.
5. At the end, **no llama-server may be running on the box**. Verify and say so in RESULT.md.
6. Do not push to any git repo. Do not edit production configs.

## Baseline (production, measured 2026-11-03)
Production env and args are in `/srv/glm-5.3/pf-test.sh` (the `prod` config: `BASE` env plus
`LLAMA_EXPERT_STORE_MAX_BATCH=2048`, args `--ctx-size 524288 --batch-size 2048 --ubatch-size 2048 --moe-cache-mib 260000 ...`).
The client `/srv/glm-5.3/pf_client.py <tag>` sends a 24k cold prompt, two cached steps (+300 and +60 tokens), a 100k
cold prompt and the standard request (md5; production gate `c592f33e7c`). Edit its URL port if you change ports.
Known: 24k prompt ~570-584 tok/s, 100k ~500 tok/s, peak HBM 88.9 GB of 97.9 at ubatch 2048 with the 16 GiB HBM tier.
Profile: half of shallow prefill time is the routed-expert MMQ reading the experts **in place from host memory**
(the cache region in Grace LPDDR5X) at ~137 GB/s effective, a third of the C2C link (decode reads reach 350-475 GB/s).
Each 2048-token ubatch reads ~3.1 GB of experts per MoE layer.

## Lever 1: bigger ubatch (config only; do this first)
Each expert byte read from host serves ubatch tokens, so 4096 and 8192 should cut the expert-read cost per token.
Legs (same BASE env, `--batch-size` = `--ubatch-size` = U, and `LLAMA_EXPERT_STORE_MAX_BATCH=U` so the in-place path
is kept): U = 2048 (baseline, rerun for a same-day number), 4096, 8192. Keep `--ctx-size 524288`. The compute buffer
grows with U; if a leg fails to fit in HBM, retry it with `LLAMA_EXPERT_STORE_HBM_MIB=8192`, then `=0`, and report
which tier it needed (the HBM tier costs decode speed: 16 GiB tier ~26-28 tok/s, off ~22.8).
Record per leg: load time, the client's JSON lines, peak GPU memory (`nvidia-smi` sampled every 2 s), and the
`compute buffer size` log lines. Note whether the standard-request md5 stays `c592f33e7c` (it should: the prompt is
short) and whether the 24k/100k outputs' md5s change between U values (they may; report, do not judge).

## Lever 2: faster expert reads (investigate, measure the ceiling, prototype only if time allows)
Why does MMQ reading host-resident weights get ~137 GB/s when the link does 350-475? Measure, with small standalone
CUDA programs (nvcc 12.8 at /usr/local/cuda/bin on the box; `-arch=sm_90`):
- a: copy bandwidth host->HBM from (i) cudaHostRegister'd malloc memory, (ii) plain malloc'd memory (unregistered),
  (iii) cudaMallocHost, using cudaMemcpyAsync with 2-16 MB chunks, 1 and 4 streams;
- b: a simple kernel streaming-reading the same host buffers (coalesced 16-byte loads, all SMs) vs HBM.
Then look at how the expert store's cache region is allocated and whether it is registered
(`~/llama-pf/src/llama-moe-cache.cpp`: `store_region_alloc`, `cudaHostRegister`, `bank_from_ptr`,
`LLAMA_EXPERT_STORE_PREFILL_FROM_CACHE` = the earlier failed attempt, patch 0044, whose staged copies ran at only
~10 GB/s). State the likely cause of 137 GB/s and of 0044's 10 GB/s, with your measurements, and the best
achievable host->HBM rate. If you have >12 minutes left after that, prototype the fix (for example: per MoE layer,
copy the used experts into an HBM staging buffer with large async copies on a second stream, double-buffered one
layer ahead) in a COPY of the tree (`cp -a ~/llama-pf ~/llama-pfub`, rebuild with
`cmake --build build --target llama-server -j 32`) and measure it with the client. Otherwise, write the design.

## RESULT.md format
1. One table: leg, U, HBM tier, load s, 24k tok/s, step-300 s, 100k tok/s, std md5, peak HBM.
2. Lever 2: a table of the bandwidth measurements, the diagnosis (3-6 sentences), and either prototype numbers or
   the design (bullets).
3. Anything that failed, with the shortest decisive log line.
4. Confirmation that no llama-server runs and nvidia-smi shows the GPU idle.
