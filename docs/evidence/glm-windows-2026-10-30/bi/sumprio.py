import sys,json
for l in open(sys.argv[1], errors='replace'):
    l=l.strip()
    if l.startswith("{"):
        d=json.loads(l)
        if "scenario" in d:
            print(d["run"], d["scenario"], d["tag"], d["spec"], "send+%s"%d.get("t_send"), "ttft %s s"%d.get("ttft_s"), "wall %s"%d.get("wall_s"), d.get("decode_tps"), d.get("md5", "ERR%s"%d.get("error")))
    elif l.startswith("===") or "FAIL" in l or "died" in l or "DONE" in l or "WATCHDOG" in l or "lowest" in l or "LEAK" in l or "no key" in l: print(l[:220])
