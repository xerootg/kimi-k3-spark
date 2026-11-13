#!/usr/bin/env python3
"""GH200 window nc10 (2026-11-13): vision requests next to text requests, with client classes. One invocation = one
timeline of requests; every request is one JSON line. Keys come from the class file and are never printed.
Usage: vis10.py NAME --classes FILE EVENT [EVENT ...] [--rounds R]
  EVENT = SECONDS:CLIENT:SPEC[:N]   CLIENT = juno | hef | wb0 | wb1 | cc (the five production clients)
  SPEC  = pN or LK (text, the requests of ekbench.py through /completion, N = tokens to generate), or
          loc-ID / read-ID / full-ID (an image written by visgen.py, sent the way the poster pipeline sends it:
          /v1/chat/completions, temperature 0, seed 1, reasoning_budget_tokens 0, cache_prompt false; loc and full use
          the locate prompt with its JSON schema and 400 tokens, read the read prompt with 800 tokens)
  --rounds R repeats the timeline R times, one after another (for requests alone: all events at second 0 would run
  together, so "alone" is NAME solo with --seq, which runs the events one after another)."""
import argparse, base64, hashlib, json, sys, threading, time, urllib.request, urllib.error
sys.path.insert(0, '/srv/glm-5.3-flash/nc')
import prio_nc
V = "/srv/glm-5.3-flash/nc/vis"
LOCATE = ("This photo shows a pole or wall with posters. Find every event poster that is mostly visible. Output only JSON: "
          "a list of {\"title\": short title, \"box\": [x0, y0, x1, y1]} with coordinates normalised to 0-1000 of the image width/height.")
READ = ("Read the event poster(s) in this image. For each poster give JSON with keys: title, dates, times, venue, address, "
        "price, urls, performers, other_text. Copy text exactly as printed; write null where absent. Output only JSON.")
SCHEMA = {"type": "json_schema", "json_schema": {"name": "posters", "schema": {"type": "array", "items": {"type": "object",
          "properties": {"title": {"type": "string"}, "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}},
          "required": ["title", "box"]}}}}
IMG = {}


def image(spec):
    if spec not in IMG:
        IMG[spec] = base64.b64encode(open("%s/%s.jpg" % (V, spec), "rb").read()).decode()
    return IMG[spec]


def vision(url, spec, key, out, client, t_start):
    read = spec.startswith("read-")
    b = {"messages": [{"role": "user", "content": [{"type": "text", "text": READ if read else LOCATE},
                                                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image(spec)}}]}],
         "temperature": 0, "seed": 1, "max_tokens": 800 if read else 400, "cache_prompt": False, "reasoning_budget_tokens": 0}
    if not read:
        b["response_format"] = SCHEMA
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    row = {"tag": client, "spec": spec, "t_send": round(t0 - t_start, 2)}
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    except urllib.error.HTTPError as e:
        row.update(error=e.code, body=e.read().decode(errors="replace")[:200]); out.append(row); return
    except Exception as e:  # connection refused or reset: the server is gone
        row.update(error="exc", body=repr(e)[:200]); out.append(row); return
    c = d["choices"][0]; text = c["message"].get("content") or ""; t = d.get("timings", {}); u = d.get("usage", {})
    row.update(wall_s=round(time.time() - t0, 2), md5=hashlib.md5(text.encode()).hexdigest()[:10], chars=len(text),
               prompt_n=u.get("prompt_tokens"), pred_n=u.get("completion_tokens"), finish=c.get("finish_reason"),
               prefill_s=round(t.get("prompt_ms", 0) / 1e3, 2), decode_tps=round(t.get("predicted_per_second", 0), 2),
               head=text.replace("\n", " ")[:60])
    out.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("events", nargs="+"); ap.add_argument("--classes", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18104"); ap.add_argument("--tag", default="")
    ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--seq", action="store_true")
    a = ap.parse_args()
    k = prio_nc.load_keys(a.classes)
    keys = {"juno": k["juno"][0], "hef": k["hef-posters"][0], "wb0": k["workbench"][0], "wb1": k["workbench"][1], "cc": k["claude-code"][0]}
    for rnd in range(a.rounds):
        out = []; th = []; T = time.time()

        def one(client, spec, n):
            if spec[0] in "pL" and spec[1:].isdigit():
                loc = []
                prio_nc.run(a.url, spec, keys[client], n, loc, client, T)
                for r in loc:
                    r["n"] = n or (256 if spec[0] == "p" else 128)
                out.extend(loc)
            else:
                vision(a.url, spec, keys[client], out, client, T)
        for ev in a.events:
            f = ev.split(":"); delay, client, spec = float(f[0]), f[1], f[2]; n = int(f[3]) if len(f) > 3 else 0
            if a.seq:
                one(client, spec, n)
            else:
                def g(delay=delay, client=client, spec=spec, n=n):
                    time.sleep(delay)
                    try:
                        one(client, spec, n)
                    except Exception as e:
                        out.append({"tag": client, "spec": spec, "t_send": delay, "error": "exc", "body": repr(e)[:200]})
                t = threading.Thread(target=g); t.start(); th.append(t)
        for t in th:
            t.join()
        for r in (out if a.seq else sorted(out, key=lambda r: r.get("t_send", 0))):
            r["scenario"] = a.name; r["kind"] = "vis"; r["leg"] = a.tag; r["round"] = rnd
            print(json.dumps(r), flush=True)
        if any(r.get("error") == "exc" for r in out):
            break


if __name__ == "__main__":
    main()
