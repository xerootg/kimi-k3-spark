#!/usr/bin/env python3
"""speedcmp.py results.jsonl TAGA TAGB [LEGS_DIR]: two legs, row by row: decode speed (solo) or aggregate speed
(several clients) of A and B, B's change in percent, B's accepted/generated drafts, and whether the answers' md5
match. Rows are paired by what the client asked (kind, prompts, temperature, seed, round; the n-th repeat of A with
the n-th of B), so the two legs may have run different client lists: rows without a partner are counted and left
out. With LEGS_DIR (holding mem-<TAG>.txt: "epoch MiB watts MemAvailable" every 2 s) also
the mean GPU power over each row's wall time and tokens per second per watt. The last lines are the means per kind
of row (solo / n clients, temperature)."""
import json, sys
from collections import defaultdict

def load(path):
    rows = defaultdict(list)
    for l in open(path):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        rows[r.get("tag")].append(r)
    return rows

def power(legs, tag):
    out = []
    if legs is None:
        return out
    try:
        for l in open("%s/mem-%s.txt" % (legs, tag)):
            f = l.split()
            if len(f) >= 3:
                out.append((float(f[0]), float(f[2])))
    except OSError:
        pass
    return out

def watts(p, r):
    w = [x for t, x in p if r["t0"] - 1 <= t <= r["t1"] + 1]
    return sum(w) / len(w) if w else None

rows = load(sys.argv[1])
ta, tb = sys.argv[2], sys.argv[3]
legs = sys.argv[4] if len(sys.argv) > 4 else None
a, b = rows.get(ta, []), rows.get(tb, [])
pa, pb = power(legs, ta), power(legs, tb)
print("%s %d rows, %s %d rows" % (ta, len(a), tb, len(b)))
def ident(r):
    return (r.get("kind"), r.get("spec") or tuple(r.get("specs", [])), r.get("temp"), r.get("seed"), r.get("round"))

byb = defaultdict(list)
for y in b:
    byb[ident(y)].append(y)
pairs = [(x, byb[ident(x)].pop(0)) for x in a if byb[ident(x)]]
if len(pairs) != len(a) or len(pairs) != len(b):
    print("paired %d rows; without a partner: %d of %s, %d of %s" % (len(pairs), len(a) - len(pairs), ta, len(b) - len(pairs), tb))
groups = defaultdict(list)
for x, y in pairs:
    solo = x["kind"] == "solo"
    if not solo and not ("agg_tps" in x and "agg_tps" in y):
        continue
    sx, sy = (x["decode_tps"], y["decode_tps"]) if solo else (x["agg_tps"], y["agg_tps"])
    if solo:
        what = "solo %-3s" % x["spec"]
        key = "solo, temperature %s" % x["temp"]
        ok, n = y.get("draft_ok") or 0, y.get("draft_n") or 0
    else:
        what = "%d clients %s" % (x.get("n", 0), x.get("round"))
        key = "%d clients, temperature %s" % (x.get("n", 0), x["temp"])
        ok, n = sum(v or 0 for v in y.get("draft_ok", [])), sum(v or 0 for v in y.get("draft_n", []))
    wx, wy = watts(pa, x), watts(pb, y)
    pw = "  %3.0f W -> %3.0f W, %.3f -> %.3f tok/s/W" % (wx, wy, sx / wx, sy / wy) if wx and wy else ""
    same = "same bytes" if x.get("md5") == y.get("md5") else "OTHER BYTES"
    print("%-16s t=%-3s seed %-3s %6.2f -> %6.2f tok/s %+6.1f %%  drafts %d/%d%s  %s" % (what, x["temp"], x["seed"], sx, sy, 100 * (sy / sx - 1), ok, n, pw, same))
    if not (not solo and x.get("round") == "warm"):
        groups[key].append((sx, sy, wx, wy, same == "same bytes"))
print("means (rounds marked warm left out):")
for key, v in groups.items():
    mx, my = sum(r[0] for r in v) / len(v), sum(r[1] for r in v) / len(v)
    pw = ""
    if all(r[2] and r[3] for r in v):
        ex, ey = sum(r[0] / r[2] for r in v) / len(v), sum(r[1] / r[3] for r in v) / len(v)
        pw = ", %.3f -> %.3f tok/s/W" % (ex, ey)
    print("  %-28s %2d rows  %6.2f -> %6.2f tok/s %+6.1f %%%s%s" % (key, len(v), mx, my, 100 * (my / mx - 1), pw, "" if all(r[4] for r in v) else "  OTHER BYTES in some row"))
