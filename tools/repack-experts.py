#!/usr/bin/env python3
"""Build an expert store from a (split) GGUF.

The store puts the three routed-expert matrices of one expert (gate, up, down)
back to back in ONE record, so an engine that streams experts from NVMe reads
one contiguous, 4 KiB-aligned record per selected expert instead of three
slices from three tensors gigabytes apart. llama.cpp patch 0003 in this repo
serves MUL_MAT_ID from these records with O_DIRECT reads into a host-RAM LRU.

Layout (little-endian, all offsets absolute in the store file):

  header, 64 bytes at offset 0
    char[4] magic = "K3XS"
    u32     version = 1 or 2
    u32     n_records        number of MoE layers in the store
    u32     n_expert         experts per layer
    u32     align            slot alignment (4096 in v1, requested in v2)
    u32     entry_size       bytes per index entry (64)
    u64     index_offset     (4096)
    u64     records_offset   first record
    u8[24]  reserved
  index: n_records entries of 64 bytes at index_offset, ascending layer
    u32 layer, u32 gate_type, u32 up_type, u32 down_type   (ggml type ids)
    u64 gate_size, u64 up_size, u64 down_size              (bytes per expert)
    u64 record_size                                        (padded to 4096)
    u64 base_offset                                        (record of expert 0)
    u8[8] reserved in v1; u64 stride in v2
  records: expert e of a layer is at base_offset + e * record_size in v1 or
    base_offset + e * stride in v2. The v2 stride rounds record_size up to
    align; all layer base offsets are align-aligned.
    gate bytes at 0, up bytes at gate_size, down bytes at gate_size + up_size,
      zero padding to record_size, then (in v2) zero padding to stride. Rows
      are packed exactly as in the GGUF.

Only stdlib: the Spark has no numpy. Reads the shards with preadv straight
into a page-aligned buffer and writes records with O_DIRECT, so a 545 GB
repack does not churn the page cache of a box that is serving the model.

usage: repack-experts.py --gguf <first shard> --out <store> [--index-only]
                         [--verify N] [--layers A-B] [--stride-align BYTES]
"""
import argparse
import json
import mmap
import os
import random
import re
import struct
import sys
import time

# ggml type id -> (block size in elements, bytes per block)
GGML_TYPES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17),
}
TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
              16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 21: "IQ3_S",
              22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16"}

HEADER = struct.Struct("<4sIIIIIQQ24x")
ENTRY = struct.Struct("<IIIIQQQQQ8x")
ENTRY_V2 = struct.Struct("<IIIIQQQQQQ")
assert HEADER.size == 64 and ENTRY.size == 64
assert ENTRY_V2.size == 64
MAGIC = b"K3XS"
ALIGN = 4096
EXPS = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def stride_alignment(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if value < ALIGN or value & (value - 1):
        raise argparse.ArgumentTypeError(
            f"must be a power of two and at least {ALIGN} bytes")
    return value


def round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def rd(f, fmt):
    s = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, f.read(s))


def rstr(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}


def rval(f, t):
    if t == 8:
        return rstr(f)
    if t == 9:
        (et,) = rd(f, "I")
        (n,) = rd(f, "Q")
        return [rval(f, et) for _ in range(n)]
    return rd(f, SCALAR[t])[0]


def parse_shard(path):
    """-> list of (name, dims, type, absolute data offset, n_bytes)"""
    out = []
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        ver, nt, nkv = rd(f, "IQQ")
        align = 32
        for _ in range(nkv):
            k = rstr(f)
            (t,) = rd(f, "I")
            v = rval(f, t)
            if k == "general.alignment":
                align = v
        infos = []
        for _ in range(nt):
            name = rstr(f)
            (nd,) = rd(f, "I")
            dims = rd(f, "Q" * nd)
            (ty,) = rd(f, "I")
            (off,) = rd(f, "Q")
            infos.append((name, dims, ty, off))
        data_start = (f.tell() + align - 1) // align * align
    for name, dims, ty, off in infos:
        if ty not in GGML_TYPES:
            raise SystemExit(f"{name}: unknown ggml type {ty}")
        bs, ts = GGML_TYPES[ty]
        n = 1
        for d in dims:
            n *= d
        if dims[0] % bs:
            raise SystemExit(f"{name}: ne0 {dims[0]} not a multiple of block size {bs}")
        n_bytes = n // bs * ts
        out.append((name, dims, ty, data_start + off, n_bytes))
    return out


def shards_of(first):
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", first)
    if not m:
        return [first]
    base, _, total = m.groups()
    return [f"{base}-{i:05d}-of-{total}.gguf" for i in range(1, int(total) + 1)]


def build_index(shards, stride_align=ALIGN):
    tensors = {}
    for sp in shards:
        for name, dims, ty, off, nb in parse_shard(sp):
            tensors[name] = (sp, dims, ty, off, nb)
    layers = {}
    for name, (sp, dims, ty, off, nb) in tensors.items():
        m = EXPS.match(name)
        if not m:
            continue
        il, which = int(m.group(1)), m.group(2)
        layers.setdefault(il, {})[which] = (sp, dims, ty, off, nb)
    entries = []
    for il in sorted(layers):
        t = layers[il]
        if set(t) != {"gate", "up", "down"}:
            raise SystemExit(f"layer {il}: incomplete expert set {sorted(t)}")
        n_expert = {t[w][1][2] for w in t}
        if len(n_expert) != 1:
            raise SystemExit(f"layer {il}: expert counts differ {n_expert}")
        n_expert = n_expert.pop()
        sizes = {w: t[w][4] // n_expert for w in t}
        for w in t:
            if t[w][4] % n_expert:
                raise SystemExit(f"layer {il} {w}: bytes not divisible by n_expert")
        total = sizes["gate"] + sizes["up"] + sizes["down"]
        record = round_up(total, ALIGN)
        entries.append({
            "layer": il, "n_expert": n_expert,
            "gate": t["gate"], "up": t["up"], "down": t["down"],
            "gate_size": sizes["gate"], "up_size": sizes["up"], "down_size": sizes["down"],
            "record_size": record,
        })
        if stride_align > ALIGN:
            entries[-1]["stride"] = round_up(record, stride_align)
    if not entries:
        raise SystemExit("no ffn_*_exps tensors found")
    n_expert = {e["n_expert"] for e in entries}
    if len(n_expert) != 1:
        raise SystemExit(f"expert count differs across layers: {n_expert}")
    index_offset = ALIGN
    records_offset = round_up(index_offset + len(entries) * ENTRY.size, stride_align)
    off = records_offset
    for e in entries:
        e["base_offset"] = off
        off += e.get("stride", e["record_size"]) * e["n_expert"]
    return entries, n_expert.pop(), index_offset, records_offset, off


def write_header_index(fd, entries, n_expert, index_offset, records_offset, stride_align):
    buf = mmap.mmap(-1, records_offset)
    version = 2 if stride_align > ALIGN else 1
    buf[0:HEADER.size] = HEADER.pack(MAGIC, version, len(entries), n_expert, stride_align, ENTRY.size,
                                    index_offset, records_offset)
    for i, e in enumerate(entries):
        p = index_offset + i * ENTRY.size
        fields = (e["layer"], e["gate"][2], e["up"][2], e["down"][2],
                  e["gate_size"], e["up_size"], e["down_size"],
                  e["record_size"], e["base_offset"])
        if version == 2:
            buf[p:p + ENTRY.size] = ENTRY_V2.pack(*fields, e["stride"])
        else:
            buf[p:p + ENTRY.size] = ENTRY.pack(*fields)
    os.pwrite(fd, buf, 0)
    buf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="first shard of the GGUF")
    ap.add_argument("--out", required=True, help="store file to write")
    ap.add_argument("--index-only", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--verify", type=int, default=0, help="after writing, compare N random records with the GGUF")
    ap.add_argument("--layers", default=None, help="A-B: only (re)write these layers' records (resume)")
    ap.add_argument("--stride-align", type=stride_alignment, default=ALIGN, metavar="BYTES",
                    help="power-of-two slot alignment, at least 4096 (default: 4096)")
    args = ap.parse_args()

    shards = shards_of(args.gguf)
    for s in shards:
        if not os.path.exists(s):
            raise SystemExit(f"missing shard {s}")
    entries, n_expert, index_offset, records_offset, total = build_index(shards, args.stride_align)
    print(f"{len(entries)} MoE layers x {n_expert} experts, store = {total/1e9:.2f} GB, "
          f"records {entries[0]['record_size']}..{max(e['record_size'] for e in entries)} bytes")
    for e in entries[:3] + entries[-1:]:
        stride = e.get("stride", e["record_size"])
        stride_text = f", stride {stride} B" if stride != e["record_size"] else ""
        print(f"  blk.{e['layer']:2d}: gate {TYPE_NAMES.get(e['gate'][2], e['gate'][2])} {e['gate_size']} B, "
              f"up {TYPE_NAMES.get(e['up'][2], e['up'][2])} {e['up_size']} B, "
              f"down {TYPE_NAMES.get(e['down'][2], e['down'][2])} {e['down_size']} B -> record {e['record_size']} B"
              f"{stride_text}")
    with open(args.out + ".json", "w") as f:
        json.dump({"n_expert": n_expert, "records_offset": records_offset, "total_bytes": total,
                   "layers": [{k: (v if not isinstance(v, tuple) else {"shard": os.path.basename(v[0]), "dims": v[1], "type": v[2], "offset": v[3], "n_bytes": v[4]})
                               for k, v in e.items()} for e in entries]}, f, indent=1)
    if args.index_only:
        return

    sel = None
    if args.layers:
        a, b = args.layers.split("-")
        sel = set(range(int(a), int(b) + 1))

    fds = {s: os.open(s, os.O_RDONLY) for s in shards}
    out = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
    if sel is None:
        os.ftruncate(out, total)
        write_header_index(out, entries, n_expert, index_offset, records_offset, args.stride_align)
    max_rec = max(e.get("stride", e["record_size"]) for e in entries)
    buf = mmap.mmap(-1, max_rec)
    mv = memoryview(buf)
    t0 = time.time()
    done = 0
    for e in entries:
        if sel is not None and e["layer"] not in sel:
            continue
        g, u, d = e["gate"], e["up"], e["down"]
        gs, us, ds, rs = e["gate_size"], e["up_size"], e["down_size"], e["record_size"]
        stride = e.get("stride", rs)
        mv[gs + us + ds:rs] = b"\0" * (rs - gs - us - ds)
        mv[rs:stride] = b"\0" * (stride - rs)
        tl = time.time()
        for x in range(n_expert):
            os.preadv(fds[g[0]], [mv[0:gs]], g[3] + x * gs)
            os.preadv(fds[u[0]], [mv[gs:gs + us]], u[3] + x * us)
            os.preadv(fds[d[0]], [mv[gs + us:gs + us + ds]], d[3] + x * ds)
            n = os.pwrite(out, mv[0:stride], e["base_offset"] + x * stride)
            if n != stride:
                raise SystemExit(f"short write {n} != {stride} at layer {e['layer']} expert {x}")
        for fd in fds.values():
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        done += stride * n_expert
        dt = time.time() - tl
        print(f"blk.{e['layer']:2d} done: {stride*n_expert/1e9:.2f} GB in {dt:.0f} s ({stride*n_expert/1e9/dt:.2f} GB/s), "
              f"total {done/1e9:.0f} GB, elapsed {(time.time()-t0)/60:.0f} min", flush=True)
    os.fsync(out)
    os.close(out)
    mv.release()
    buf.close()

    if args.verify:
        print(f"verifying {args.verify} random records")
        sfd = os.open(args.out, os.O_RDONLY)
        bad = 0
        for _ in range(args.verify):
            e = random.choice(entries)
            x = random.randrange(n_expert)
            stride = e.get("stride", e["record_size"])
            rec = os.pread(sfd, stride, e["base_offset"] + x * stride)
            pos = 0
            for w in ("gate", "up", "down"):
                t = e[w]
                sz = e[w + "_size"]
                ref = os.pread(fds[t[0]], sz, t[3] + x * sz)
                if rec[pos:pos + sz] != ref:
                    bad += 1
                    print(f"MISMATCH layer {e['layer']} expert {x} {w}")
                pos += sz
            if any(rec[pos:]):
                bad += 1
                print(f"PADDING NOT ZERO layer {e['layer']} expert {x}")
        print("verify:", "OK" if bad == 0 else f"{bad} problems")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
