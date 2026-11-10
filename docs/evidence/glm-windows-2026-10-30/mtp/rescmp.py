# compare the client-side results of two legs: same commands in the same order, md5 and token counts must match
import json, sys
a, b = sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
def key(r):
    return (r.get("md5") if not isinstance(r.get("md5"), list) else tuple(r["md5"]), r.get("decode_n", r.get("gen")), r.get("stop"))
ra = [r for r in rows if r["tag"] == a]; rb = [r for r in rows if r["tag"] == b]
n = min(len(ra), len(rb)); bad = 0
for i in range(n):
    if key(ra[i]) != key(rb[i]) or ra[i]["kind"] != rb[i]["kind"]:
        bad += 1; print("DIFF row", i, ra[i]["kind"], key(ra[i]), key(rb[i]))
print(f"{a} rows {len(ra)}, {b} rows {len(rb)}, compared {n}, different {bad}")
print("RESULTS-IDENTICAL" if bad == 0 and len(ra) == len(rb) else "RESULTS-DIFFERENT")
