#!/usr/bin/env python3
"""Vision test for GLM-5.3-Flash on llama-server: per photo, read the whole photo at full resolution and at 1536 px,
locate posters (pass A, 1536 px), and read the main poster's full-resolution crop (pass B). One JSON line per request."""
import base64, hashlib, io, json, sys, time, urllib.request
from PIL import Image
URL = "http://127.0.0.1:%s/v1/chat/completions" % (sys.argv[1] if len(sys.argv) > 1 else "18104")
D = "/srv/glm-5.3-flash/posters/"
CROPS = {"2652d15d": (510,260,2440,3640), "7d734f4f": (930,1080,2010,2600), "eaa17d7e": (1350,1520,2010,2590), "fb48f8cf": (660,120,1950,2600)}
READ = ("Read the event poster(s) in this image. For each poster give JSON with keys: title, dates, times, venue, address, "
        "price, urls, performers, other_text. Copy text exactly as printed; write null where absent. Output only JSON.")
LOCATE = ("This photo shows a pole or wall with posters. Find every event poster that is mostly visible. Output only JSON: "
          "a list of {\"title\": short title, \"box\": [x0, y0, x1, y1]} with coordinates normalised to 0-1000 of the image width/height.")
def jpeg(img, q=92):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=q); return b.getvalue()
def ask(tag, text, data, max_tokens=900):
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}}]}],
            "temperature": 0, "seed": 1, "max_tokens": max_tokens, "cache_prompt": False}
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800).read())
    t = d.get("timings", {}); c = d["choices"][0]["message"].get("content") or ""
    print(json.dumps({"tag": tag, "prompt_tokens": d["usage"]["prompt_tokens"], "prefill_ms": round(t.get("prompt_ms", 0)),
                      "decode_tps": round(t.get("predicted_per_second", 0), 1), "out_tokens": d["usage"]["completion_tokens"],
                      "wall_s": round(time.time() - t0, 1), "md5": hashlib.md5(c.encode()).hexdigest()[:10], "content": c}), flush=True)
    return c
# text-only baseline for the prompt tokens
body = {"messages": [{"role": "user", "content": READ}], "max_tokens": 1, "temperature": 0}
d = json.loads(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
print(json.dumps({"tag": "text-only READ prompt", "prompt_tokens": d["usage"]["prompt_tokens"]}), flush=True)
for k in sorted(CROPS):
    im = Image.open(D + k + "-image.jpg"); im.load()
    small = im.copy(); small.thumbnail((1536, 1536))
    ask(k + " read-full", READ, open(D + k + "-image.jpg", "rb").read())
    ask(k + " read-1536", READ, jpeg(small))
    ask(k + " locate-1536", LOCATE, jpeg(small), 400)
    ask(k + " read-crop", READ, jpeg(im.crop(CROPS[k])))
im = Image.open(D + "2652d15d-image.jpg"); ask("2652d15d read-crop repeat", READ, jpeg(im.crop(CROPS["2652d15d"])))
