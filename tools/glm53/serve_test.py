#!/usr/bin/env python3
"""serve_test.py: g53x serveidx against g53x check, bit for bit (standard library only).

  ./tools/glm53/build-sycl.sh && python3 tools/glm53/serve_test.py [--iters 200]

Host part: builds four small K3XS expert stores in $W/tmp/serve_test (W = ~/glm53-work) from the extract.py slices in
$W/data (UD-Q2_K_XL and UD-Q4_K_XL, layers 3, 8 and 75, experts 0-7; data/*.json), each in store v1 and in v2 with
--stride-align 524288 semantics (tools/repack-experts.py layout and .json sidecar), plus a layer 78 of Q2_K/Q3_K
records (zeros, like GLM's MTP layer: types g53x has no kernel for). Then runs the inner part in the oneAPI container
(dsycl.sh).

Inner part (in the container, $W at /w): `g53x <slice> TG TU TD check` writes the reference rows (.x, .dn) of experts
0-7 of every slice; then for every store a `g53x SET STORE - - serveidx PORT` is started and a client checks over TCP:
  - every served expert alone (k = 1) and several experts in one request (k = 4, 7, 8, and 16 with repeats) against the
    check rows, bit for bit, x = the check's x;
  - NaN (k rows) for: a layer not in the store, an expert in the store but not in SET, an unsupported layer (78), a
    request mixing a served and an unserved id, a request mixing two layers (k3x: one kind per request);
  - k = 0 closes the connection, and a new connection works after that;
  - K3X_SETOUT lists exactly the served experts;
  - per-request latency (send to last reply byte) for k = 1, 4, 8, median of --iters, per layer.
The v1 stores take SET as "layer expert" lines, the v2 stores as k3x IDX lines "gid tg td offset". The Q2/Q4 v1 stores
run a second time with K3X_NOSTREAM=1 (one down launch, rows sent at the end).
"""
import argparse
import json
import math
import os
import socket
import statistics
import struct
import subprocess
import sys
import time

TYPE_IDS = {"Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ4_XS": 23}
TYPE_BLK = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 17: 74, 18: 98, 23: 136}   # bytes per 256 values
QUANTS = ["UD-Q2_K_XL", "UD-Q4_K_XL"]
LAYERS = [3, 8, 75]
NE = 8
MTP = 78                     # unsupported layer: Q2_K gate/up, Q3_K down
NOT_IN_SET = (75, 7)          # in the store, left out of SET
ALIGN = 4096
HEADER = struct.Struct("<4sIIIIIQQ24x")
ENTRY = struct.Struct("<IIIIQQQQQ8x")
ENTRY_V2 = struct.Struct("<IIIIQQQQQQ")


def round_up(x, a):
    return (x + a - 1) // a * a


# ---------------------------------------------------------------- host: stores
def slices(data, quant):
    """-> {layer: {"gate"|"up"|"down": json entry}} from data/<quant>.json"""
    out = {}
    for e in json.load(open(os.path.join(data, quant + ".json"))):
        out.setdefault(e["layer"], {})[e["proj"]] = e
    return out


def build_store(data, quant, path, stride_align):
    sl = slices(data, quant)
    layers = []
    for il in LAYERS:
        t = sl[il]
        assert {t[w]["n_expert"] for w in t} == {NE}
        layers.append({"layer": il, "src": t, "types": [TYPE_IDS[t[w]["type"]] for w in ("gate", "up", "down")],
                       "dims": {w: [t[w]["ne0"], t[w]["ne1"], NE] for w in t},
                       "sizes": [t[w]["bytes_per_expert"] for w in ("gate", "up", "down")]})
    D, F = layers[0]["dims"]["gate"][0], layers[0]["dims"]["gate"][1]
    layers.append({"layer": MTP, "src": None, "types": [10, 10, 11],
                   "dims": {"gate": [D, F, NE], "up": [D, F, NE], "down": [F, D, NE]},
                   "sizes": [F * D // 256 * 84, F * D // 256 * 84, D * F // 256 * 110]})
    version = 2 if stride_align > ALIGN else 1
    records_offset = round_up(ALIGN + len(layers) * ENTRY.size, stride_align)
    off = records_offset
    for L in layers:
        L["record_size"] = round_up(sum(L["sizes"]), ALIGN)
        L["stride"] = round_up(L["record_size"], stride_align) if version == 2 else L["record_size"]
        L["base_offset"] = off
        off += L["stride"] * NE
    total = off
    with open(path, "wb") as f:
        f.truncate(total)
        f.write(HEADER.pack(b"K3XS", version, len(layers), NE, stride_align, ENTRY.size, ALIGN, records_offset))
        for i, L in enumerate(layers):
            fields = (L["layer"], *L["types"], *L["sizes"], L["record_size"], L["base_offset"])
            f.seek(ALIGN + i * ENTRY.size)
            f.write(ENTRY_V2.pack(*fields, L["stride"]) if version == 2 else ENTRY.pack(*fields))
        for L in layers:
            if not L["src"]:
                continue   # MTP stand-in: zeros (file hole)
            blobs = [open(os.path.join(data, L["src"][w]["file"]), "rb").read() for w in ("gate", "up", "down")]
            for e in range(NE):
                f.seek(L["base_offset"] + e * L["stride"])
                for b, n in zip(blobs, L["sizes"]):
                    assert len(b) == n * NE
                    f.write(b[e * n:(e + 1) * n])
    with open(path + ".json", "w") as f:   # the repack-experts.py sidecar (the fields serveidx reads, and the rest)
        json.dump({"n_expert": NE, "records_offset": records_offset, "total_bytes": total, "layers": [
            {"layer": L["layer"], "n_expert": NE,
             **{w: {"shard": (L["src"][w]["file"] if L["src"] else ""), "dims": L["dims"][w], "type": L["types"][i],
                    "offset": 0, "n_bytes": L["sizes"][i] * NE} for i, w in enumerate(("gate", "up", "down"))},
             "gate_size": L["sizes"][0], "up_size": L["sizes"][1], "down_size": L["sizes"][2],
             "record_size": L["record_size"], "base_offset": L["base_offset"],
             **({"stride": L["stride"]} if version == 2 else {})} for L in layers]}, f, indent=1)
    return layers


def write_set(path, layers, idx_format):
    with open(path, "w") as f:
        for L in layers:
            for e in range(NE):
                if (L["layer"], e) == NOT_IN_SET:
                    continue
                if idx_format:
                    f.write(f"{L['layer'] * NE + e} {L['types'][0]} {L['types'][2]} {L['base_offset'] + e * L['stride']}\n")
                else:
                    f.write(f"{L['layer']} {e}\n")


def host(args):
    T = os.path.dirname(os.path.abspath(__file__))
    W = os.path.expanduser(os.environ.get("W", "~/glm53-work"))
    out = os.path.join(W, "tmp", "serve_test")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for q in QUANTS:
        for v, sa in (("v1", ALIGN), ("v2", 524288)):
            name = f"{q}.{v}.k3xs"
            layers = build_store(os.path.join(W, "data"), q, os.path.join(out, name), sa)
            write_set(os.path.join(out, name + ".set"), layers, idx_format=(v == "v2"))
    print(f"stores built in {time.time() - t0:.1f} s ({out})", flush=True)
    cmd = f"python3 /w/tools/serve_test.py inner --iters {args.iters}" + (f" --only {args.only}" if args.only else "")
    return subprocess.run([os.path.join(T, "dsycl.sh"), cmd]).returncode


# ---------------------------------------------------------------- inner: server + client
def recv_all(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError(f"closed after {len(b)} of {n} bytes")
        b += c
    return bytes(b)


class Client:
    def __init__(self, port, D):
        self.port, self.D = port, D
        self.connect()

    def connect(self):
        self.s = socket.create_connection(("127.0.0.1", self.port))
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def request(self, gids, x, seq=0):
        k = len(gids)
        hdr = [k] + list(gids) + [0] * (16 - k)
        if k < 16:
            hdr[16] = seq
        self.s.sendall(struct.pack("<17i", *hdr) + x)
        return recv_all(self.s, k * self.D * 4)


def all_nan(b):
    return all(math.isnan(v) for v in struct.unpack(f"<{len(b) // 4}f", b))


def run_store(args, g53x, store, quant, ref, env_extra, label):
    D = 6144
    port = free_port()
    log = store + f".{label}.log"
    setout = store + f".{label}.setout"
    env = dict(os.environ, K3X_SETOUT=setout, **env_extra)
    if quant == "UD-Q4_K_XL":
        env["G53X_EXACT"] = "0"  # Q4 regression exercises the retained non-exact path
    with open(log, "w") as lf:
        srv = subprocess.Popen([g53x, store + ".set", store, "-", "-", "serveidx", str(port)], stderr=lf, stdout=lf, env=env)
    try:
        t0 = time.time()
        while True:
            txt = open(log).read()
            if "serving on" in txt:
                break
            if srv.poll() is not None or time.time() - t0 > 300:
                raise SystemExit(f"{label}: server did not start:\n{txt}")
            time.sleep(0.2)
        for line in txt.splitlines():
            if line.startswith(("loaded", "layer", "self-check")):
                print(f"  [{label}] {line}")
        fails = []
        check = lambda ok, what: ok or fails.append(what)
        c = Client(port, D)
        served = [(L, e) for L in LAYERS for e in range(NE) if (L, e) != NOT_IN_SET]
        x = ref[(quant, LAYERS[0])][0]
        nrows = 0
        # every served expert alone
        for L, e in served:
            xs, dn = ref[(quant, L)]
            r = c.request([L * NE + e], xs, seq=nrows)
            check(r == dn[e * D * 4:(e + 1) * D * 4], f"k=1 L{L} e{e} differs from check")
            nrows += 1
        # several experts per request, in shuffled orders, and k = 16 with repeats (hdr[16] is then an id, no seq)
        for L in LAYERS:
            xs, dn = ref[(quant, L)]
            es = [e for e in range(NE) if (L, e) != NOT_IN_SET]
            orders = [es[::-1], [es[i] for i in (2, 0, 3, 1)], es[1::2] + es[0::2], (es * 3)[:16]]
            for o in orders:
                r = c.request([L * NE + e for e in o], xs)
                want = b"".join(dn[e * D * 4:(e + 1) * D * 4] for e in o)
                check(r == want, f"k={len(o)} L{L} {o} differs from check")
                nrows += len(o)
        # NaN replies (and the connection keeps working after them)
        nan_cases = {
            "layer not in the store": [5 * NE + 0],
            "expert in the store, not in SET": [NOT_IN_SET[0] * NE + NOT_IN_SET[1]],
            "unsupported layer (Q2_K/Q3_K)": [MTP * NE + 1],
            "served + unserved id": [3 * NE + 0, 3 * NE + 1, 5 * NE + 2],
            "two layers in one request": [3 * NE + 0, 8 * NE + 0],
            "negative id": [-1],
        }
        for what, gids in nan_cases.items():
            r = c.request(gids, x)
            check(len(r) == len(gids) * D * 4 and all_nan(r), f"NaN reply expected: {what}")
            xs, dn = ref[(quant, 8)]
            check(c.request([8 * NE + 4], xs) == dn[4 * D * 4:5 * D * 4], f"row after NaN reply ({what}) differs")
        # k = 0 closes the connection; a new one works
        c.s.sendall(struct.pack("<17i", *([0] * 17)) + x)
        try:
            closed = c.s.recv(1) == b""
        except ConnectionError:
            closed = True
        check(closed, "k=0 did not close the connection")
        c.s.close()
        c.connect()
        xs, dn = ref[(quant, 75)]
        check(c.request([75 * NE + 6, 75 * NE + 0], xs) == dn[6 * D * 4:7 * D * 4] + dn[0:D * 4], "after reconnect")
        # K3X_SETOUT
        got = sorted(tuple(map(int, l.split())) for l in open(setout))
        check(got == sorted(served), f"K3X_SETOUT lists {len(got)} experts, {len(served)} served")
        # latency
        lat = {}
        for L in LAYERS:
            xs, _ = ref[(quant, L)]
            es = [e for e in range(NE) if (L, e) != NOT_IN_SET]
            for k in (1, 4, 8):   # L75 serves 7 experts: its k = 8 repeats one
                ts = []
                for i in range(args.iters + 10):
                    o = [es[(i * 3 + j) % len(es)] for j in range(k)]
                    t0 = time.perf_counter_ns()
                    r = c.request([L * NE + e for e in o], xs)
                    t1 = time.perf_counter_ns()
                    if i >= 10:
                        ts.append((t1 - t0) / 1e3)
                lat[(L, k)] = statistics.median(ts)
        c.s.close()
        time.sleep(0.2)
        srv.terminate()
        srv.wait(10)
        print(f"  [{label}] {nrows} rows compared, {len(nan_cases)} NaN cases, latency median of {args.iters} (us): " +
              "  ".join(f"L{L} " + "/".join(f"{lat[(L, k)]:.0f}" for k in (1, 4, 8) if (L, k) in lat) for L in LAYERS) +
              "  (k=1/4/8)")
        for f in fails:
            print(f"  [{label}] FAIL: {f}")
        return fails, lat
    finally:
        if srv.poll() is None:
            srv.kill()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def inner(args):
    os.chdir("/w")
    g53x = "/w/g53x"
    out = "/w/tmp/serve_test"
    dumps = os.path.join(out, "dump")
    os.makedirs(dumps, exist_ok=True)
    ref = {}
    for q in QUANTS:
        sl = slices("/w/data", q)
        for L in LAYERS:
            ty = [str(TYPE_IDS[sl[L][w]["type"]]) for w in ("gate", "up", "down")]
            name = f"{q}_L{L}"
            env = dict(os.environ)
            if q == "UD-Q4_K_XL":
                env["G53X_EXACT"] = "0"
            p = subprocess.run([g53x, f"data/{name}", *ty, "check", os.path.join(dumps, name)], capture_output=True, text=True, env=env)
            if p.returncode:
                raise SystemExit(f"check {name} failed:\n{p.stderr}")
            fused = [l for l in p.stderr.splitlines() if "fused" in l]
            print(f"check {name} exact={env.get('G53X_EXACT', '1')} {' '.join(ty)}: {fused[0] if fused else ''}", flush=True)
            xs = open(os.path.join(dumps, name + ".x"), "rb").read()
            dn = open(os.path.join(dumps, name + ".dn"), "rb").read()
            assert len(xs) == 6144 * 4 and len(dn) == NE * 6144 * 4
            ref[(q, L)] = (xs, dn)
    allfails, table = [], {}
    runs = [(q, v, {}, f"{v}") for q in QUANTS for v in ("v1", "v2")]
    runs += [(q, "v1", {"K3X_NOSTREAM": "1"}, "v1-nostream") for q in QUANTS]
    unknown = set(filter(None, args.only.split(","))) - {f"{q}.{label}" for q, _, _, label in runs}
    if unknown:
        raise SystemExit(f"--only: no such run {sorted(unknown)}")
    for q, v, env, label in runs:
        if args.only and f"{q}.{label}" not in args.only.split(","):
            continue
        print(f"{q} store {v}{' ' + str(env) if env else ''}:", flush=True)
        fails, lat = run_store(args, g53x, os.path.join(out, f"{q}.{v}.k3xs"), q, ref, env, label)
        allfails += [f"{q} {label}: {f}" for f in fails]
        table[(q, label)] = lat
    print("\nper-request latency, client send to last byte, median of %d (us)" % args.iters)
    print("%-11s %-12s %s" % ("quant", "server", "  ".join(f"L{L:<2} k=1/4/8 " for L in LAYERS)))
    for (q, label), lat in table.items():
        print("%-11s %-12s %s" % (q, label, "  ".join(
            "/".join(f"{lat[(L, k)]:.0f}" for k in (1, 4, 8) if (L, k) in lat).ljust(13) for L in LAYERS)))
    if allfails:
        print(f"\nFAILED: {len(allfails)}")
        for f in allfails:
            print("  " + f)
        return 1
    print("\nPASS: serveidx rows bit-identical to g53x check (UD-Q2_K_XL, UD-Q4_K_XL; store v1, v2; streamed and not), "
          "NaN replies as k3x")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="host", choices=["host", "inner"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--only", default="", help="comma list of QUANT.LABEL runs, e.g. UD-Q4_K_XL.v1,UD-Q2_K_XL.v1-nostream")
    args = ap.parse_args()
    return host(args) if args.mode == "host" else inner(args)


if __name__ == "__main__":
    sys.exit(main())
