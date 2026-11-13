#!/usr/bin/env python3
"""GH200 window nc10: writes the request images for vis10.py once, on the CPU, so that every leg sends the same bytes.
Per photo (the four of the vision PR, 3000x4000): loc-ID.jpg = letterbox to 1000x1000 (the poster pipeline's locate
request), read-ID.jpg = the poster crop capped at 1.2 megapixels (its read request), full-ID.jpg = the photo as it is
(12 megapixels, the server caps it at 6.3; the pipeline never sends this, it is the longest image prompt)."""
import hashlib, os, shutil
from PIL import Image
SRC = "/srv/glm-5.3-flash/posters/%s-image.jpg"; OUT = "/srv/glm-5.3-flash/nc/vis"
CROPS = {"2652d15d": (510, 260, 2440, 3640), "7d734f4f": (930, 1080, 2010, 2600), "eaa17d7e": (1350, 1520, 2010, 2590),
         "fb48f8cf": (660, 120, 1950, 2600)}
os.makedirs(OUT, exist_ok=True)


def letterbox(im, n=1000):
    w, h = im.size; f = n / max(w, h); r = im.resize((round(w * f), round(h * f)), Image.BICUBIC)
    c = Image.new("RGB", (n, n), (0, 0, 0)); c.paste(r, ((n - r.size[0]) // 2, (n - r.size[1]) // 2)); return c


def cap(im, px):
    w, h = im.size
    if w * h <= px:
        return im
    f = (px / (w * h)) ** 0.5
    return im.resize((round(w * f), round(h * f)), Image.BICUBIC)


for k, box in CROPS.items():
    im = Image.open(SRC % k); im.load(); im = im.convert("RGB")
    letterbox(im).save("%s/loc-%s.jpg" % (OUT, k[:4]), "JPEG", quality=92)
    cap(im.crop(box), 1.2e6).save("%s/read-%s.jpg" % (OUT, k[:4]), "JPEG", quality=92)
    shutil.copyfile(SRC % k, "%s/full-%s.jpg" % (OUT, k[:4]))
for f in sorted(os.listdir(OUT)):
    b = open(os.path.join(OUT, f), "rb").read()
    print(f, Image.open(os.path.join(OUT, f)).size, len(b), hashlib.md5(b).hexdigest()[:10])
