#!/usr/bin/env python3
"""Compare legs of a window nd results.jsonl: per (kind, n, round) the md5 list of each tag against the reference tag,
and throughput. usage: ndcmp.py results.jsonl REFTAG TAG..."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
ref, tags = sys.argv[2], sys.argv[3:]
def key(r): return (r["kind"], r.get("n", 1), str(r.get("round", r.get("p"))))
by = {}
for r in rows: by.setdefault(r["tag"], {})[key(r)] = r
for t in tags:
    same = diff = 0
    for k, r in sorted(by.get(t, {}).items()):
        rr = by.get(ref, {}).get(k)
        if rr is None: continue
        m = r["md5"] if isinstance(r["md5"], list) else [r["md5"]]
        mr = rr["md5"] if isinstance(rr["md5"], list) else [rr["md5"]]
        eq = m == mr
        same += eq; diff += not eq
        tps = r.get("agg_tps", r.get("decode_tps")); tpr = rr.get("agg_tps", rr.get("decode_tps"))
        print(f"{t:6s} {k[0]:5s} n={k[1]} round={k[2]:4s} {'SAME' if eq else 'DIFF'} tok/s {tps:6.2f} vs {ref} {tpr:6.2f} ({100*(tps/tpr-1):+.1f}%) md5 {m}")
    print(f"{t}: {same} measurements byte-identical to {ref}, {diff} differ")
