#!/usr/bin/env python3
"""ekbench.py of the nc windows (/srv/glm-5.3-flash/ekbench.py) plus `--key-file PATH`: the API key in that file (one
line, generated on the box, never printed) goes into the Authorization header of every request, so the client is a
member of a client class when the server runs with LLAMA_CLIENT_CLASSES (a request without a key gets 401 then).
Nothing else changed. Request specs: pN = mbench prompt N (256 tokens), LK = bench.py long prompt of ~K tokens
(seed 7, 128 tokens; L16384 is the 16k gold request). Every request: temperature 0, seed 1, ignore_eos,
cache_prompt false.
  solo  SPECS...        : each spec alone, one after another
  multi SPECS... --rounds R : all specs together (barrier start), one warm round + R measured rounds
One JSON line per measurement.
"""
import argparse, hashlib, json, random, sys, threading, time, urllib.request, urllib.error

PROMPTS = [
    "Write a Python quicksort function and explain it.",
    "Explain how a Kubernetes operator reconciles desired and actual state, with a short Go example.",
    "Write a short story about a lighthouse keeper who finds a stranded robot on the rocks.",
    "Summarize the main causes of the French Revolution in five bullet points, then discuss which mattered most.",
    "Describe the differences between TCP and UDP and when to use each, with examples from games and video calls.",
    "Write a haiku sequence about autumn in Seattle, then explain the imagery.",
    "Explain the math behind RSA encryption step by step with a toy example.",
    "Give a recipe for sourdough bread and explain the role of each ingredient.",
]
WORDS = ("kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap "
         "helm chart argocd sync health probe container image registry runner pipeline postgres backup barman wal "
         "archive snapshot etcd raft leader follower kubelet cgroup memory cpu throttle oom eviction taint toleration "
         "affinity scheduler controller operator webhook certificate issuer acme dns").split()


def long_prompt(tokens, seed=7):
    rnd = random.Random(seed)
    return "Summarize this log:\n" + " ".join(rnd.choice(WORDS) for _ in range(int(tokens / 1.29)))


CACHE = False
KEY = None


def body(spec, n_predict):
    if spec[0] == "p":
        return {"prompt": PROMPTS[int(spec[1:])], "temperature": 0, "seed": 1, "n_predict": n_predict or 256,
                "cache_prompt": CACHE, "ignore_eos": True}
    return {"prompt": long_prompt(int(spec[1:])), "temperature": 0, "seed": 1, "n_predict": n_predict or 128,
            "cache_prompt": CACHE, "ignore_eos": True}


def post(url, b, timeout=7200):
    hdr = {"Content-Type": "application/json"}
    if KEY:
        hdr["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(url, data=json.dumps(b).encode(), headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:300]}


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()[:10]


def one(url, spec, n_predict):
    s = time.time()
    d = post(url + "/completion", body(spec, n_predict))
    e = time.time()
    if "error" in d:
        return {"spec": spec, "error": d["error"], "body": d["body"], "t0": s, "t1": e}
    t = d["timings"]
    return {"spec": spec, "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n"), "prefill_s": round(t["prompt_ms"] / 1e3, 2), "prefill_tps": round(t["prompt_per_second"], 2),
            "decode_n": t["predicted_n"], "decode_tps": round(t["predicted_per_second"], 2), "md5": md5(d["content"]),
            "t0": s, "t1": e}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["solo", "multi"])
    p.add_argument("specs", nargs="+")
    p.add_argument("--url", default="http://127.0.0.1:18080")
    p.add_argument("--tag", default="")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--n-predict", type=int, default=0)
    p.add_argument("--cache", action="store_true", help="cache_prompt true (default false)")
    p.add_argument("--key-file", default="", help="file holding the API key to send (never printed)")
    a = p.parse_args()
    global CACHE, KEY
    CACHE = a.cache
    if a.key_file:
        KEY = open(a.key_file).readline().strip()
    if a.mode == "solo":
        for spec in a.specs:
            r = one(a.url, spec, a.n_predict)
            r.update({"kind": "solo", "tag": a.tag})
            print(json.dumps(r), flush=True)
        return
    for rnd in range(a.rounds + 1):
        res = [None] * len(a.specs)
        bar = threading.Barrier(len(a.specs))

        def w(i):
            bar.wait()
            res[i] = one(a.url, a.specs[i], a.n_predict)

        th = [threading.Thread(target=w, args=(i,)) for i in range(len(a.specs))]
        [x.start() for x in th]
        [x.join() for x in th]
        st = min(r["t0"] for r in res)
        en = max(r["t1"] for r in res)
        gen = sum(r.get("decode_n", 0) for r in res)
        print(json.dumps({"kind": "multi", "tag": a.tag, "n": len(a.specs), "round": "warm" if rnd == 0 else rnd,
                          "agg_tps": round(gen / (en - st), 2), "wall_s": round(en - st, 2), "gen": gen,
                          "per_stream_tps": [r.get("decode_tps") for r in res], "md5": [r.get("md5", "ERR%s" % r.get("error")) for r in res],
                          "specs": a.specs, "prefill_tps": [r.get("prefill_tps") for r in res], "t0": st, "t1": en}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
