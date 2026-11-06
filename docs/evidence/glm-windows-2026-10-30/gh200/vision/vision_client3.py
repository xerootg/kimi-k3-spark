#!/usr/bin/env python3
"""Live endpoint tests with reasoning_budget_tokens=0: locate at 1024 px for all 4 photos, read 1.2 MP crops 3x each
(reproducibility), the native 8k-token crop 3x. Key on stdin. usage: vision_client3.py BASEURL < key"""
import base64, hashlib, io, json, sys, time, urllib.request
from PIL import Image
BASE = sys.argv[1]; KEY = sys.stdin.readline().strip()
D = "/srv/glm-5.3-flash/posters/"
CROPS = {"2652d15d": (510,260,2440,3640), "7d734f4f": (930,1080,2010,2600), "eaa17d7e": (1350,1520,2010,2590), "fb48f8cf": (660,120,1950,2600)}
READ = ("Read the event poster(s) in this image. For each poster give JSON with keys: title, dates, times, venue, address, "
        "price, urls, performers, other_text. Copy text exactly as printed; write null where absent. Output only JSON.")
LOCATE = ("This photo shows a pole or wall with posters. Find every event poster that is mostly visible. Output only JSON: "
          "a list of {\"title\": short title, \"box\": [x0, y0, x1, y1]} with coordinates normalised to 0-1000 of the image width/height.")
def jpeg(img, q=92):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=q); return b.getvalue()
def cap(img, px):
    w, h = img.size; s = (px / (w * h)) ** 0.5
    return img if s >= 1 else img.resize((int(w * s), int(h * s)), Image.BICUBIC)
def ask(tag, text, data, max_tokens=800):
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}}]}],
            "temperature": 0, "seed": 1, "max_tokens": max_tokens, "cache_prompt": False, "reasoning_budget_tokens": 0}
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    t = d.get("timings", {}); m = d["choices"][0]["message"]; c = m.get("content") or ""
    print(json.dumps({"tag": tag, "prompt_tokens": d["usage"]["prompt_tokens"], "prefill_ms": round(t.get("prompt_ms", 0)),
                      "out_tokens": d["usage"]["completion_tokens"], "wall_s": round(time.time() - t0, 1),
                      "reasoning_chars": len(m.get("reasoning_content") or ""), "md5": hashlib.md5(c.encode()).hexdigest()[:10], "content": c}), flush=True)
ims = {k: Image.open(D + k + "-image.jpg") for k in CROPS}
for im in ims.values(): im.load()
for k, im in sorted(ims.items()):
    s = im.copy(); s.thumbnail((1024, 1024)); ask(k + " locate-1024", LOCATE, jpeg(s), 400)
for k, im in sorted(ims.items()):
    c = jpeg(cap(im.crop(CROPS[k]), 1.2e6))
    for i in range(3): ask("%s read-1.2MP #%d" % (k, i), READ, c)
c = jpeg(ims["2652d15d"].crop(CROPS["2652d15d"]))
for i in range(3): ask("2652d15d read-native #%d" % i, READ, c)
