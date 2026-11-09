import sys,json
for l in open(sys.argv[1], errors='replace'):
    l=l.strip()
    if l.startswith("{"):
        d=json.loads(l)
        if d["kind"]=="solo": print(d["tag"],"solo",d["spec"],d.get("md5"),d.get("decode_tps"), d.get("prefill_tps"))
        else: print(d["tag"],"N=%d"%d["n"],d["round"],d["md5"],d["agg_tps"],d["per_stream_tps"])
    elif l.startswith("===") or "FAIL" in l or "died" in l or "DONE" in l or "WATCHDOG" in l or "lowest" in l: print(l[:220])
