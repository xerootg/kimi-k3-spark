#!/usr/bin/env python3
"""Past-1M check: two facts (one at the start, one in the middle) in a ~N-token log, questions at the end.
R1 reads the whole prompt (cache_prompt true) and asks for the start fact; R2 and R3 reuse the cached prefix and ask
for the middle fact and for a short description of the log (coherence, and decode speed at depth).
Usage: longctx.py URL TAG TOKENS"""
import hashlib, json, random, sys, time, urllib.request

url, tag, tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = ("kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap helm "
     "chart argocd sync health probe container image registry runner pipeline postgres backup barman wal archive").split()
r = random.Random(11)
n_words = int(tokens / 1.29)
half = n_words // 2
f1 = " ".join(r.choice(W) for _ in range(half))
f2 = " ".join(r.choice(W) for _ in range(n_words - half))
prefix = ("Remember this: the access code for vault seven is 4729-ALPHA-PLUM.\n\nLog follows:\n" + f1 +
          "\nNote: the password for locker nine is CEDAR-81-QUILL.\n" + f2 + "\n\nEnd of log. ")
Q = {"start": "Question: what is the access code for vault seven? Answer with the code only.\nAnswer:",
     "middle": "Question: what is the password for locker nine? Answer with the password only.\nAnswer:",
     "describe": "Question: in two or three sentences, describe what kinds of words make up the log above, and say what the very first line asked you to remember.\nAnswer:"}
NP = {"start": 24, "middle": 24, "describe": 160}


def post(path, body, timeout=6 * 3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


t0 = time.time()
ntok = len(post("/tokenize", {"content": prefix + Q["start"]})["tokens"])
print(json.dumps({"kind": "longctx-tokens", "tag": tag, "prompt_tokens": ntok, "tokenize_s": round(time.time() - t0, 1)}), flush=True)
for k in ("start", "middle", "describe"):
    t0 = time.time()
    d = post("/completion", {"prompt": prefix + Q[k], "temperature": 0, "seed": 1, "n_predict": NP[k], "cache_prompt": True})
    t = d["timings"]
    c = d["content"]
    print(json.dumps({"kind": "longctx", "tag": tag, "q": k, "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n"),
                      "prefill_tps": round(t["prompt_per_second"], 2), "prefill_s": round(t["prompt_ms"] / 1e3, 1),
                      "decode_n": t["predicted_n"], "decode_tps": round(t["predicted_per_second"], 2),
                      "found_start": "4729-ALPHA-PLUM" in c.upper().replace(" ", ""),
                      "found_middle": "CEDAR-81-QUILL" in c.upper().replace(" ", ""),
                      "content": c[:400], "md5": hashlib.md5(c.encode()).hexdigest()[:10], "t0": t0, "t1": time.time()}), flush=True)
