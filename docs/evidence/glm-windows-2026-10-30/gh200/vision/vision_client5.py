import base64, hashlib, io, json, sys, time, urllib.request
from PIL import Image
BASE = sys.argv[1]; KEY = sys.stdin.readline().strip()
LOCATE = ("This photo shows a pole or wall with posters. Find every event poster that is mostly visible. Output only JSON: "
          "a list of {\"title\": short title, \"box\": [x0, y0, x1, y1]} with coordinates normalised to 0-1000 of the image width/height.")
SCHEMA = {"type": "json_schema", "json_schema": {"name": "posters", "schema": {"type": "array", "items": {"type": "object",
          "properties": {"title": {"type": "string"}, "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}},
          "required": ["title", "box"]}}}}
def ask(tag, img, schema):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=92)
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": LOCATE}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}]}],
            "temperature": 0, "seed": 1, "max_tokens": 400, "cache_prompt": False, "reasoning_budget_tokens": 0}
    if schema: body["response_format"] = SCHEMA
    t0 = time.time(); d = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY}), timeout=900).read())
    print(tag, d["usage"], round(time.time() - t0, 1), "s:", (d["choices"][0]["message"].get("content") or "").replace("\n", " ")[:600])
def letterbox(im, n=1000):
    w, h = im.size; f = n / max(w, h); r = im.resize((round(w * f), round(h * f)), Image.BICUBIC)
    c = Image.new("RGB", (n, n), (0, 0, 0)); c.paste(r, ((n - r.size[0]) // 2, (n - r.size[1]) // 2)); return c, f, (n - r.size[0]) // 2, (n - r.size[1]) // 2
for k in ("7d734f4f", "2652d15d", "fb48f8cf"):
    im = Image.open("/srv/glm-5.3-flash/posters/%s-image.jpg" % k); im.load(); lb, f, ox, oy = letterbox(im)
    print(k, "scale", round(f, 4), "offset", ox, oy); ask(k + " letterbox-1000", lb, True)
