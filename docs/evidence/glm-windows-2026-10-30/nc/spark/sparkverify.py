import json, sys, hashlib, time, urllib.request
# glm53-spark production check through the kimi-k3 Service: quicksort, temperature 0, seed 1, 128 tokens, cache_prompt
# false. Gold since #172: ee079496a4 (p0 256 tokens: ce2fc95752). Key on stdin (never printed). Usage: sparkverify.py N
key = sys.stdin.readline().strip()
url = "http://10.43.61.128:8080/completion"
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
    body = {"prompt": "Write a Python quicksort function and explain it.", "temperature": 0, "seed": 1, "n_predict": 128, "cache_prompt": False}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    print(json.dumps({"round": i, "tps": round(d["timings"]["predicted_per_second"], 2), "md5": hashlib.md5(d["content"].encode()).hexdigest()[:10], "wall_s": round(time.time() - t0, 2)}), flush=True)
