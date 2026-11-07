#!/usr/bin/env python3
"""Window nd ownership: box gets experts [x0, x1) of every regular Flash layer (IQ2_XS/IQ2_XS/IQ3_XXS).
Writes <name>.set (engine: "layer expert"), <name>.idx (f53s: "gid offset" into <name>.bin) and <name>.dd
(shell lines that stream the records in that order with O_DIRECT)."""
import json, sys
store, name, x0, x1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = json.load(open(store + ".json"))
ne = d["n_expert"]
off = 0
with open(name + ".set", "w") as fs, open(name + ".idx", "w") as fi, open(name + ".dd", "w") as fd:
    for l in d["layers"]:
        if (l["gate"]["type"], l["up"]["type"], l["down"]["type"]) != (17, 17, 18):
            continue
        L, R, B = l["layer"], l["record_size"], l["base_offset"]
        assert R == 8060928
        for x in range(x0, x1):
            fs.write(f"{L} {x}\n")
            fi.write(f"{L*ne + x} {off}\n")
            off += R
        fd.write(f"dd if={store} iflag=direct,skip_bytes bs={R} skip={B + x0*R} count={x1-x0} status=none\n")
print(name, "records", off // 8060928, "bytes", off)
