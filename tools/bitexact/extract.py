# Extract a few expert records of one layer from the K3 expert store with small preads.
# usage: extract.py STORE LAYER OUT "e1,e2,..."
import json, os, sys
store, layer, out, ids = sys.argv[1], int(sys.argv[2]), sys.argv[3], [int(v) for v in sys.argv[4].split(",")]
j = json.load(open(store + ".json"))
L = next(l for l in j["layers"] if l["layer"] == layer)
fd = os.open(store, os.O_RDONLY)
with open(out, "wb") as f:
    for e in ids:
        off = L["base_offset"] + e * L["record_size"]
        b = os.pread(fd, L["record_size"], off)
        assert len(b) == L["record_size"]
        f.write(b)
        os.posix_fadvise(fd, off, L["record_size"], os.POSIX_FADV_DONTNEED)
print(layer, L["gate"]["type"], L["up"]["type"], L["down"]["type"], L["record_size"], len(ids))
