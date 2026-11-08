#!/usr/bin/env python3
"""Per-op decode GPU time from a llama-server log run with GGML_CUDA_OP_PROFILE=N (kimi-k3-spark patch 0003 profiler,
KV-tagged keys from patch 0051). For each request (split at launch_slot_ 'processing task'), averages the profile blocks whose node
time per graph is below --max-ms (decode graphs; prompt graphs take seconds) and prints the KV and indexer rows plus
the total. Usage: opprof.py LOG [LOG ...] [--max-ms 400] [--labels 16k,100k]"""
import re, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--max-ms", type=float, default=400.0)
ap.add_argument("--labels", default="")
ap.add_argument("--all", action="store_true", help="print every row, not only KV/indexer ones")
a = ap.parse_args()
labels = a.labels.split(",") if a.labels else []
head = re.compile(r"op profile over (\d+) graphs: ([\d.]+) ms of node time per graph")
row = re.compile(r"^\s*([\d.]+) ms/graph\s+([\d.]+)%\s+([\d.]+) us/call\s+(\d+) calls/graph\s+(.*)$")
INTEREST = ("kv=", "LIGHTNING_INDEXER", "FLASH_ATTN_EXT", "TOP_K", "ARGSORT")

for log in a.logs:
    reqs = []  # list of lists of blocks; block = (total_ms, {key: ms})
    cur_block = None
    for line in open(log, errors="replace"):
        line = line.rstrip("\n")
        body = line.split(" ", 2)[-1] if line[:1].isdigit() else line
        if "launch_slot_" in line and "processing task" in line:
            reqs.append([])
            cur_block = None
            continue
        m = head.search(line)
        if m:
            cur_block = (float(m.group(2)), {})
            if reqs:
                reqs[-1].append(cur_block)
            continue
        if cur_block is not None:
            r = row.match(body.strip()) or row.match(line.split(" I ")[-1].strip() if " I " in line else "")
            if r:
                cur_block[1][r.group(5).strip()] = float(r.group(1))
            elif "ms/graph" not in line:
                cur_block = None
    print(f"== {log}")
    for i, blocks in enumerate(reqs):
        dec = [b for b in blocks if b[0] < a.max_ms]
        if not dec:
            continue
        lab = labels[i] if i < len(labels) else f"request {i}"
        # skip the first decode block of a request (it may include the end of the prompt)
        use = dec[1:] if len(dec) > 1 else dec
        tot = sum(b[0] for b in use) / len(use)
        keys = sorted({k for b in use for k in b[1]})
        avg = {k: sum(b[1].get(k, 0.0) for b in use) / len(use) for k in keys}
        print(f"  {lab}: {len(use)} decode blocks, {tot:.2f} ms of node time per graph")
        for k, v in sorted(avg.items(), key=lambda kv: -kv[1]):
            if a.all or any(s in k for s in INTEREST):
                print(f"    {v:7.3f} ms/graph  {k}")
