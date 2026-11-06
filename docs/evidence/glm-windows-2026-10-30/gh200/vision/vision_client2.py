#!/usr/bin/env python3
"""Vision window 2: thinking off; reproducibility with encoder embedding dumps; pass-A boxes at 1536 and 1024;
pass-B crops at native size vs capped to 2 MP and 1 MP. One JSON line per request."""
import base64, hashlib, io, json, os, shutil, sys, time, urllib.request
from PIL import Image
URL = "http://127.0.0.1:%s/v1/chat/completions" % sys.argv[1]
EMB = sys.argv[2]  # MTMD_DEBUG_EMBEDDINGS path the server writes
D = "/srv/glm-5.3-flash/posters/"
CROPS = {"2652d15d": (510,260,2440,3640), "7d734f4f": (930,1080,2010,2600), "eaa17d7e": (1350,1520,2010,2590), "fb48f8cf": (660,120,1950,2600)}
READ = ("Read the event poster(s) in this image. For each poster give JSON with keys: title, dates, times, venue, address, "
        "price, urls, performers, other_text. Copy text exactly as printed; write null where absent. Output only JSON.")
LOCATE = ("This photo shows a pole or wall with posters. Find every event poster that is mostly visible. Output only JSON: "
          "a list of {\"title\": short title, \"box\": [x0, y0, x1, y1]} with coordinates normalised to 0-1000 of the image width/height.")
def jpeg(img, q=92):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=q); return b.getvalue()
def cap(img, mp):
    w, h = img.size; s = (mp * 1e6 / (w * h)) ** 0.5
    return img if s >= 1 else img.resize((int(w * s), int(h * s)), Image.BICUBIC)
def ask(tag, text, data, max_tokens=900):
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}}]}],
            "temperature": 0, "seed": 1, "max_tokens": max_tokens, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800).read())
    t = d.get("timings", {}); m = d["choices"][0]["message"]; c = m.get("content") or ""
    emb = hashlib.md5(open(EMB, "rb").read()).hexdigest()[:10] if os.path.exists(EMB) else None
    print(json.dumps({"tag": tag, "prompt_tokens": d["usage"]["prompt_tokens"], "prefill_ms": round(t.get("prompt_ms", 0)),
                      "out_tokens": d["usage"]["completion_tokens"], "wall_s": round(time.time() - t0, 1),
                      "reasoning_chars": len(m.get("reasoning_content") or ""), "emb_md5": emb,
                      "md5": hashlib.md5(c.encode()).hexdigest()[:10], "content": c}), flush=True)
ims = {k: Image.open(D + k + "-image.jpg") for k in CROPS}
for im in ims.values(): im.load()
# reproducibility: the same crop three times, embeddings dumped each time
for i in range(3):
    ask("repro 2652d15d crop #%d" % i, READ, jpeg(ims["2652d15d"].crop(CROPS["2652d15d"])))
for i in range(2):
    ask("repro 7d734f4f crop #%d" % i, READ, jpeg(ims["7d734f4f"].crop(CROPS["7d734f4f"])))
for k, im in sorted(ims.items()):
    for side in (1536, 1024):
        s = im.copy(); s.thumbnail((side, side)); ask("%s locate-%d" % (k, side), LOCATE, jpeg(s), 500)
    cr = im.crop(CROPS[k])
    for mp in (99, 2.0, 1.0):
        ask("%s read-crop-%s" % (k, "native" if mp == 99 else "%.0fMP" % mp), READ, jpeg(cap(cr, mp)))
