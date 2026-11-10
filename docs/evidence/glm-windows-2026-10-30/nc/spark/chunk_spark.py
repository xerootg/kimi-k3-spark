#!/usr/bin/env python3
"""No-compromise window nc1: do a prompt's bytes depend on its ubatch boundaries? Run with LLAMA_LOGITS_FNV=1 on the
server (the prompt's last-ubatch logits row is hashed into the log). Long prompts by token ids (ekbench long_prompt,
tokenized once by the server), temperature 0, seed 1, ignore_eos.
  cold SPEC N        : the whole prompt, cache_prompt false, N tokens generated
  split SPEC CUT N   : tokens[:CUT] with cache_prompt true and 1 token generated, then the whole prompt with
                       cache_prompt true (reuses CUT cells: the rest runs in ubatches that start at CUT), N tokens
One JSON line per request (md5 of the generated text, prompt_n, cache_n)."""
import argparse, hashlib, json, sys, time, urllib.request, urllib.error
sys.path.insert(0, '/srv/glm-5.3-flash')
import ekbench


def post(url, path, b):
    req = urllib.request.Request(url + path, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:300]}


def comp(url, toks, n, cache, tag, kind):
    t0 = time.time()
    d = post(url, "/completion", {"prompt": toks, "temperature": 0, "seed": 1, "n_predict": n, "cache_prompt": cache, "ignore_eos": True})
    if "error" in d:
        print(json.dumps({"kind": kind, "tag": tag, "error": d}), flush=True); return
    t = d["timings"]
    print(json.dumps({"kind": kind, "tag": tag, "n_tok": len(toks), "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n"),
                      "prefill_tps": round(t["prompt_per_second"], 1), "md5": hashlib.md5(d["content"].encode()).hexdigest()[:10],
                      "t0": t0, "t1": time.time()}), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["cold", "split"]); ap.add_argument("spec"); ap.add_argument("args", nargs="*", type=int)
ap.add_argument("--url", default="http://127.0.0.1:18080"); ap.add_argument("--tag", default="")
a = ap.parse_args()
text = ekbench.long_prompt(int(a.spec[1:]))
toks = post(a.url, "/tokenize", {"content": text, "add_special": True})["tokens"]
if a.mode == "cold":
    comp(a.url, toks, a.args[0], False, a.tag, "chunk-cold")
else:
    cut, n = a.args
    comp(a.url, toks[:cut], 1, True, a.tag, "chunk-prefix")
    comp(a.url, toks, n, True, a.tag, "chunk-split")
