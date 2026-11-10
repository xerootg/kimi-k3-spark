#!/usr/bin/env python3
"""Digest for the Spark windows nc-s2/nc-s3. Usage: acc2.py results.jsonl TAG [TAG...]
ekbench rows: solo md5 against the Spark golds, every multi round against solo/gold, speeds.
fb2 rows: per label every md5 seen (one value = the client's bytes never changed), refusals, speeds.
prio rows: one line per request."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
GOLD = {"p0": "ce2fc95752", "p1": "bb64203d71", "p2": "2d592efcb6", "p3": "1aab74269d", "p4": "0c3f28294b", "L16384": "145c14aa8b"}
for tag in sys.argv[2:]:
    R = [r for r in rows if r.get("tag") == tag or r.get("leg") == tag]
    print(f"== {tag} ({len(R)} rows)")
    solo = {}; bad = 0; lab = {}
    for r in R:
        k = r.get("kind")
        if k == "solo" and "md5" in r:
            solo.setdefault(r["spec"], set()).add(r["md5"])
            g = GOLD.get(r["spec"]); n = r.get("n_predict") or r.get("decode_n")
            print(f"  solo {r['spec']:7s} {r['md5']} {'gold' if r['md5'] == g else ''} dec {r['decode_tps']:6.2f} pre {r['prefill_tps']:8.1f} n {n}")
        elif k == "multi":
            ok = all(m == GOLD.get(s, next(iter(solo.get(s, {'?'})))) for s, m in zip(r["specs"], r["md5"]))
            bad += not ok
            print(f"  multi n={r['n']} round={r['round']} agg {r['agg_tps']:6.2f} per {r['per_stream_tps']} {'OK' if ok else 'DIFF'} {r['specs']} {r['md5']}")
        elif k == "prio":
            if "error" in r:
                print(f"  {r['scenario']:11s} {r['tag']:6s} {r['spec']:7s} error {r['error']}")
            else:
                print(f"  {r['scenario']:11s} {r['tag']:6s} {r['spec']:7s} send {r['t_send']:6.2f} ttft {r['ttft_s']:8.3f} wall {r['wall_s']:7.2f} gap {r['max_gap_s']:6.2f} dec {r['decode_tps']:6.2f} pre {r['prefill_tps']:7.1f} {r['md5']}")
        elif "label" in r:
            if "error" in r:
                print(f"  {k:18s} {r['label']:9s} n_tok {r['n_tok']:7d} ERROR {r['error']} after {r['wall_s']} s {str(r.get('body'))[:150]}")
            else:
                if r.get("decode_n", 0) >= 8:
                    lab.setdefault(r["label"], {}).setdefault(r["md5"], []).append(k)
                print(f"  {k:18s} {r['label']:9s} rnd {str(r.get('round', '')):2s} slot {r['id_slot']} new {r['prompt_n']:7d} cached {r['cache_n']:7d} pre {r['prefill_s']:7.1f} s {r['prefill_tps']:7.1f} dec {r['decode_tps']:6.2f} wall {r['wall_s']:7.1f} {r['md5']}")
    for s, ms in solo.items():
        if s in GOLD and ms != {GOLD[s]}:
            print(f"  SOLO-DIFF {s} {ms} gold {GOLD[s]}")
    if any(r.get("kind") == "multi" for r in R):
        print(f"  multi rounds differing from solo/gold: {bad}")
    for l, ms in lab.items():
        print(f"  label {l:9s}: {'ONE md5' if len(ms) == 1 else 'DIFFERENT md5s'} " + ", ".join(f"{m} x{len(v)}" for m, v in ms.items()))
