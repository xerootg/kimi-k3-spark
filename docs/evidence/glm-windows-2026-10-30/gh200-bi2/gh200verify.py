import json, sys, hashlib, time, urllib.request
# The production gold check through the glm53-gh200 Service: quicksort, temperature 0, seed 1, 128 tokens, cache_prompt
# false (gold.py). Key on stdin (never printed). Usage: python3 gh200verify.py N
key = sys.stdin.readline().strip()
url = "http://10.43.143.254:8080/completion"
body = {"prompt": "Write a Python quicksort function and explain it.", "temperature": 0, "seed": 1, "n_predict": 128, "cache_prompt": False}
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    print(json.dumps({"round": i, "tps": round(d["timings"]["predicted_per_second"], 2), "md5": hashlib.md5(d["content"].encode()).hexdigest()[:10], "wall_s": round(time.time() - t0, 2)}), flush=True)
