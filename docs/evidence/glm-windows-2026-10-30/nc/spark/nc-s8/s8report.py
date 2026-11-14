#!/usr/bin/env python3
"""s8report.py RESULTS.jsonl : window nc-s8 digest. Solo legs S<point>.<round>: pass 1 = the first five records
(p0-p4), pass 2 = the next five; per leg the mean decode tok/s and prompt tok/s of each pass and the md5 check against
the golds of nc-s7. Per point the mean over its processes with the min-max spread, and the difference to the control
p0067. Multi legs S<point>.m: aggregate tok/s per round and the md5s."""
import json, sys, collections, statistics as st
GOLD = {"p0": "ce2fc95752", "p1": "bb64203d71", "p2": "2d592efcb6", "p3": "1aab74269d", "p4": "0c3f28294b"}
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
solo = collections.defaultdict(list); multi = collections.defaultdict(list)
for r in rows:
    (solo if r.get("kind") == "solo" else multi)[r["tag"]].append(r)
per_point = collections.defaultdict(list)
print("| leg | pass 1 decode | pass 2 decode | pass 1 prompt | pass 2 prompt | md5 |")
print("|---|---|---|---|---|---|")
for tag in sorted(solo, key=lambda t: (t.split(".") + [""])[:2]):
    rs = solo[tag]
    if any("error" in r for r in rs):
        print(f"| {tag} | ERROR {[r.get('error') for r in rs]} |"); continue
    p1, p2 = rs[:5], rs[5:10]
    bad = [r["spec"] for r in rs if GOLD.get(r["spec"]) != r["md5"]]
    d1 = st.mean(r["decode_tps"] for r in p1); d2 = st.mean(r["decode_tps"] for r in p2) if len(p2) == 5 else float("nan")
    f1 = st.mean(r["prefill_tps"] for r in p1); f2 = st.mean(r["prefill_tps"] for r in p2) if len(p2) == 5 else float("nan")
    print(f"| {tag} | {d1:.2f} | {d2:.2f} | {f1:.1f} | {f2:.1f} | {'gold' if not bad else 'DIFF ' + ','.join(bad)} |")
    per_point[tag.split(".")[0]].append((d1, d2, f1, f2))
print()
print("| point | n | pass 1 decode mean (min-max) | vs control | pass 2 decode mean (min-max) | vs control | pass 1 prompt | pass 2 prompt |")
print("|---|---|---|---|---|---|---|---|")
ctl = per_point.get("S67")
c1 = st.mean(x[0] for x in ctl) if ctl else None; c2 = st.mean(x[1] for x in ctl) if ctl else None
for pt in sorted(per_point):
    xs = per_point[pt]
    m1 = st.mean(x[0] for x in xs); m2 = st.mean(x[1] for x in xs)
    lo1, hi1 = min(x[0] for x in xs), max(x[0] for x in xs); lo2, hi2 = min(x[1] for x in xs), max(x[1] for x in xs)
    v1 = f"{100*(m1/c1-1):+.2f} %" if c1 else "-"; v2 = f"{100*(m2/c2-1):+.2f} %" if c2 else "-"
    print(f"| {pt} | {len(xs)} | {m1:.2f} ({lo1:.2f}-{hi1:.2f}) | {v1} | {m2:.2f} ({lo2:.2f}-{hi2:.2f}) | {v2} | {st.mean(x[2] for x in xs):.1f} | {st.mean(x[3] for x in xs):.1f} |")
if multi:
    print()
    print("| leg | round | aggregate tok/s | per stream | md5 |")
    print("|---|---|---|---|---|")
    for tag in sorted(multi):
        for r in multi[tag]:
            print(f"| {tag} | {r['round']} | {r['agg_tps']} | {r['per_stream_tps']} | {r['md5']} |")
