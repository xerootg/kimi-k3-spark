#!/usr/bin/env python3
"""The production Service check (vwin2 text gate): quicksort prompt, temperature 0, seed 1, 128 tokens, cache_prompt
false. Production gold on the GH200 (image g7f117e1, --parallel 1): md5 2a73cda2a3. Usage: gold.py [N] --url --tag"""
import argparse, hashlib, json, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("n", nargs="?", type=int, default=2)
ap.add_argument("--url", default="http://127.0.0.1:18104"); ap.add_argument("--tag", default="")
a = ap.parse_args()
for i in range(a.n):
    b = {"prompt": "Write a Python quicksort function and explain it.", "temperature": 0, "seed": 1, "n_predict": 128, "cache_prompt": False}
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(a.url + "/completion", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
    print(json.dumps({"kind": "gold", "tag": a.tag, "md5": hashlib.md5(d["content"].encode()).hexdigest()[:10], "decode_tps": round(d["timings"]["predicted_per_second"], 2), "wall_s": round(time.time() - t0, 2)}), flush=True)
