#!/usr/bin/env python3
"""MTP speculative decoding test client (GH200). Three request shapes, one JSON line each:
  std-t0 : standard quicksort prompt, temperature 0, 256 tokens (natural stop allowed)
  std-t1 : same prompt, temperature 1.0 / top_p 0.95, seed 1
  agent  : chat completion with a tool, like an OpenClaw step (temperature 1.0), up to 600 tokens
"""
import json, sys, time, urllib.request
URL = "http://127.0.0.1:18102"
def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.loads(urllib.request.urlopen(req, timeout=1800).read()); return d, time.time() - t0
tag = sys.argv[1]
P = "Write a Python quicksort function and explain it."
for name, body in [("std-t0", {"prompt": P, "temperature": 0, "seed": 1, "n_predict": 256, "cache_prompt": False}),
                   ("std-t1", {"prompt": P, "temperature": 1.0, "top_p": 0.95, "seed": 1, "n_predict": 256, "cache_prompt": False})]:
    d, w = post("/completion", body); t = d["timings"]
    print(json.dumps({"tag": tag, "req": name, "n": t["predicted_n"], "tps": round(t["predicted_per_second"], 2),
                      "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"), "wall": round(w, 1)}), flush=True)
tools = [{"type": "function", "function": {"name": "kubectl_get", "description": "Read Kubernetes resources",
          "parameters": {"type": "object", "properties": {"kind": {"type": "string"}, "namespace": {"type": "string"}, "name": {"type": "string"}}, "required": ["kind"]}}}]
msgs = [{"role": "system", "content": "You are Juno, an executive assistant who manages a home Kubernetes cluster. Be brief."},
        {"role": "user", "content": "The garage pods in namespace garage keep restarting. Check them, then tell me in two sentences what you will look at next."}]
d, w = post("/v1/chat/completions", {"messages": msgs, "tools": tools, "temperature": 1.0, "top_p": 0.95, "seed": 1, "max_tokens": 600})
t = d.get("timings", {})
print(json.dumps({"tag": tag, "req": "agent", "n": t.get("predicted_n"), "tps": round(t.get("predicted_per_second", 0), 2),
                  "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"), "wall": round(w, 1),
                  "finish": d["choices"][0].get("finish_reason")}), flush=True)
