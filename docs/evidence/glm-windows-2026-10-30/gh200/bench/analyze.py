#!/usr/bin/env python3
"""Join results.jsonl with power.csv: mean GPU and module watts per measurement window; summary JSON + table."""
import json
import statistics as st
import sys
from pathlib import Path

run = Path(sys.argv[1])
res = [json.loads(l) for l in open(run / "results.jsonl") if l.startswith("{")]
pw = []
for l in open(run / "power.csv"):
    f = l.strip().split(",")
    try:
        pw.append((float(f[0]), float(f[1]), float(f[2]), float(f[3])))
    except (ValueError, IndexError):
        pass


def watts(t0, t1):
    s = [p for p in pw if t0 <= p[0] <= t1]
    if not s:
        return None, None, None
    return round(st.mean(p[1] for p in s), 1), round(st.mean(p[2] for p in s), 1), round(st.mean(p[3] for p in s), 1)


out = {"depth": [], "conc": {}}
for r in res:
    g, m, u = watts(r["t0"], r["t1"])
    r.update(gpu_w=g, module_w=m, gpu_util=u)
    if r["kind"] == "depth":
        out["depth"].append(r)
    elif r["round"] != "warm":
        out["conc"].setdefault(r["tag"], []).append(r)

summary = []
for tag, rs in out["conc"].items():
    agg = [r["agg_tps"] for r in rs]
    mw = [r["module_w"] for r in rs if r["module_w"]]
    gw = [r["gpu_w"] for r in rs if r["gpu_w"]]
    md5 = sorted({m for r in rs for m in r["md5"]})
    per = [x for r in rs for x in r["per_stream_tps"]]
    summary.append({"tag": tag, "n": rs[0]["n"], "agg_tps_mean": round(st.mean(agg), 2), "agg_tps_max": max(agg),
                    "per_stream_mean": round(st.mean(per), 2), "module_w": round(st.mean(mw), 1) if mw else None,
                    "gpu_w": round(st.mean(gw), 1) if gw else None,
                    "tok_per_kj_module": round(1000 * st.mean(agg) / st.mean(mw), 1) if mw else None,
                    "md5s": md5, "rounds": len(rs)})
out["summary"] = summary
idle = [p for p in pw[:6]]
out["idle_sample"] = {"gpu_w": round(st.mean(p[1] for p in idle), 1), "module_w": round(st.mean(p[2] for p in idle), 1)} if idle else None
json.dump(out, open(run / "summary.json", "w"), indent=1)
print("DEPTH")
for r in out["depth"]:
    print(f"  ctx {r['prompt_n']:>6}  prefill {r['prefill_tps']:7.1f} tok/s ({r['prefill_s']:6.1f} s)  decode {r['decode_tps']:5.2f} tok/s  module {r['module_w']} W  gpu {r['gpu_w']} W")
print("CONCURRENCY")
for s in summary:
    print(f"  {s['tag']:<10} N={s['n']:>2}  agg {s['agg_tps_mean']:6.2f} (max {s['agg_tps_max']:6.2f})  per-stream {s['per_stream_mean']:5.2f}  module {s['module_w']} W  gpu {s['gpu_w']} W  tok/kJ {s['tok_per_kj_module']}  md5s {s['md5s']}")
