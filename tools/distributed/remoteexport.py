#!/usr/bin/env python3
"""Export the remote expert set's records from the expert store as one flat file plus a text index, for the
remote expert server (LLAMA_EXPERT_STORE_REMOTE).

usage: remoteexport.py STORE SET.txt INDEX_OUT [MB_PER_S]  > records.bin   (typically piped to ssh)

SET.txt: "layer expert" lines (tools/distributed/remoteset.py --out). The records go to stdout sorted by
(layer, expert), each as gate | up | down exactly as in the store, without the store's 4 KiB padding. INDEX_OUT
gets one line per record, in file order: "<global_id> <gate_type> <down_type> <byte_offset>", global_id =
layer * n_expert + expert (what the engine sends), ggml type ids (IQ1_S 19, IQ2_XXS 16; up has gate's type).
Reads use O_DIRECT so a box that is serving the model does not churn its page cache; MB_PER_S throttles.
"""
import mmap, os, struct, sys, time

def main():
    store, setf, idxf = sys.argv[1], sys.argv[2], sys.argv[3]
    rate = float(sys.argv[4]) * 1e6 if len(sys.argv) > 4 else 0
    fd = os.open(store, os.O_RDONLY | os.O_DIRECT)
    with open(store, "rb") as f:
        h = f.read(64)
        magic, ver, n_rec, n_exp, align, esz, ioff, roff = struct.unpack("<4sIIIIIQQ", h[:40])
        assert magic == b"K3XS"
        f.seek(ioff)
        ent = {}
        for _ in range(n_rec):
            e = struct.unpack("<IIIIQQQQQ", f.read(esz)[:56])
            ent[e[0]] = e  # layer, gate_t, up_t, down_t, gate_sz, up_sz, down_sz, rec_sz, base
    pairs = sorted({tuple(int(v) for v in l.split()) for l in open(setf) if l.strip()})
    buf = mmap.mmap(-1, max(e[7] for e in ent.values()))
    out = sys.stdout.buffer
    off = 0
    t0 = time.time()
    with open(idxf, "w") as idx:
        for n, (il, x) in enumerate(pairs):
            e = ent[il]
            assert e[1] == e[2], f"layer {il}: gate and up types differ"
            rec, used = e[7], e[4] + e[5] + e[6]
            got = os.preadv(fd, [memoryview(buf)[:rec]], e[8] + x * rec)
            assert got == rec, (il, x, got)
            out.write(memoryview(buf)[:used])
            idx.write(f"{il * n_exp + x} {e[1]} {e[3]} {off}\n")
            off += used
            if rate:
                ahead = off / rate - (time.time() - t0)
                if ahead > 0:
                    time.sleep(ahead)
            if n % 500 == 0:
                print(f"{n}/{len(pairs)} records, {off/1e9:.1f} GB, {off/1e6/max(time.time()-t0,1e-3):.0f} MB/s", file=sys.stderr)
    out.flush()
    print(f"done: {len(pairs)} records, {off} bytes, {time.time()-t0:.0f} s", file=sys.stderr)

main()
