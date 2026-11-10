#!/usr/bin/env python3
# Window nuc5: watts during the measured rounds. Spark GPU power from logs/power-<leg>.txt ("epoch watts", nvidia-smi
# power.draw every 2 s), NUC package power from logs/rapl-nuc{2,3}.log ("epoch energy_uj" of intel-rapl:0 every 5 s;
# package only, not the wall). Usage: nuc5power.py [dir]   (dir holds results.jsonl and logs/)
import json, sys, os, collections
d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
MAXR = 262143328850
def load(f): return [tuple(map(float, l.split())) for l in open(f) if len(l.split()) == 2]
rapl = [load(os.path.join(d, "logs", "rapl-nuc%d.log" % n)) for n in (2, 3)]
def nuc_w(rows, a, b):
    seg = [r for r in rows if a - 5 <= r[0] <= b + 5]
    if len(seg) < 2: return float("nan")
    e = sum((y[1] - x[1]) % MAXR for x, y in zip(seg, seg[1:]))
    return e / 1e6 / (seg[-1][0] - seg[0][0])
out = collections.defaultdict(list)
for l in open(os.path.join(d, "results.jsonl")):
    r = json.loads(l)
    if r.get("kind") != "multi" or r.get("round") == "warm" or "t0" not in r: continue
    mix = r["specs"][0].startswith("L")
    f = os.path.join(d, "logs", "power-%s.txt" % r["tag"])
    if not os.path.exists(f): continue
    g = [w for t, w in load(f) if r["t0"] <= t <= r["t1"]]
    if not g: continue
    out[(("mix" if mix else "N=%d" % r["n"]), r["tag"])].append((r["agg_tps"], sum(g) / len(g), nuc_w(rapl[0], r["t0"], r["t1"]), nuc_w(rapl[1], r["t0"], r["t1"])))
print("%-6s %-5s %7s %9s %7s %7s %14s" % ("load", "leg", "tok/s", "Spark GPU", "nuc2", "nuc3", "tok/s per W*"))
for (n, tag), v in sorted(out.items()):
    m = [sum(x[i] for x in v) / len(v) for i in range(4)]
    print("%-6s %-5s %7.2f %8.1fW %6.1fW %6.1fW %14.3f" % (n, tag, m[0], m[1], m[2], m[3], m[0] / (m[1] + m[2] + m[3])))
print("* over Spark GPU + both NUC packages, idle NUCs included (about 6.6 + 7.8 W)")
