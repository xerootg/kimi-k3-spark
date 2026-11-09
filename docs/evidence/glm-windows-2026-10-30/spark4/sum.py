import sys, json
for l in sys.stdin:
    l = l.strip()
    if not l.startswith("{"):
        print(l); continue
    d = json.loads(l)
    if d["scenario"] == "multi-agg":
        print("  AGG r%s n%s %s tok/s %s" % (d["round"], d["n"], d["agg_tps"], d["md5"])); continue
    print("  %-7s %-14s %-3s send %6s ttft %7s wall %7s pn %6s pf %6s dec %6s %s %s" % (d.get("tag"), d.get("spec"), d.get("round", ""), d.get("t_send"), d.get("ttft_s"), d.get("wall_s"), d.get("prompt_n"), d.get("prefill_tps"), d.get("decode_tps"), d.get("md5"), d.get("error", "")))
