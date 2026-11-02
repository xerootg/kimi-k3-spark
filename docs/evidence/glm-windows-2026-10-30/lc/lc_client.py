#!/usr/bin/env python3
"""Client for window lc (Spark GLM at 128k context). One JSON line per request on stdout.

  std   NAME N           : the standard request N times (gold md5 9449ae5738... at ctx 65536)
  depth NAME L1,L2,...   : one request per prompt length (cache off), 64 forced tokens; prefill and decode tok/s
  reuse NAME L           : a prompt of ~L tokens (cache on), then the same prompt plus ~300 new tokens (cache on):
                           the second request should process only the new tokens (per-turn cost O(new tokens))
"""
import hashlib
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:18080/completion"
STD = {"prompt": "Write a Python quicksort function and explain it.", "temperature": 0, "seed": 1, "n_predict": 128,
       "cache_prompt": False}
WORDS = ("kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap "
         "helm chart argocd sync health probe container image registry runner pipeline postgres backup barman wal "
         "archive snapshot etcd raft leader follower kubelet cgroup memory cpu throttle oom eviction taint toleration "
         "affinity scheduler controller operator webhook certificate issuer acme dns").split()


def post(body, timeout=7200):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, t0, time.time()


def text(tokens, seed=7):
    rnd = random.Random(seed)
    return "Summarize this log:\n" + " ".join(rnd.choice(WORDS) for _ in range(int(tokens / 1.29)))


def emit(name, kind, d, t0, t1, **extra):
    t = d["timings"]
    print(json.dumps({"leg": name, "kind": kind, "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n"),
                      "prefill_tps": round(t["prompt_per_second"], 2), "prefill_s": round(t["prompt_ms"] / 1e3, 1),
                      "decode_n": t["predicted_n"], "decode_tps": round(t["predicted_per_second"], 3),
                      "md5": hashlib.md5(d["content"].encode()).hexdigest(), "wall_s": round(t1 - t0, 1), **extra}),
          flush=True)


def main():
    mode, name, arg = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "std":
        for i in range(int(arg)):
            d, t0, t1 = post(STD)
            emit(name, f"std{i + 1}", d, t0, t1)
    elif mode == "depth":
        for L in [int(x) for x in arg.split(",")]:
            d, t0, t1 = post({"prompt": text(L), "temperature": 0, "seed": 1, "n_predict": 64, "cache_prompt": False,
                              "ignore_eos": True})
            emit(name, f"depth{L}", d, t0, t1, target=L)
    elif mode == "reuse":
        L = int(arg)
        base = text(L)
        d, t0, t1 = post({"prompt": base, "temperature": 0, "seed": 1, "n_predict": 32, "cache_prompt": True,
                          "ignore_eos": True})
        emit(name, "reuse-first", d, t0, t1, target=L)
        more = base + "\nFollow-up: " + text(300, seed=11)[len("Summarize this log:\n"):]
        d, t0, t1 = post({"prompt": more, "temperature": 0, "seed": 1, "n_predict": 32, "cache_prompt": True,
                          "ignore_eos": True})
        emit(name, "reuse-followup", d, t0, t1, target=L)
    else:
        raise SystemExit("unknown mode")


if __name__ == "__main__":
    main()
