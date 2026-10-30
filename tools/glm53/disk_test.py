#!/usr/bin/env python3
"""disk_test.py: the g53x serveidx disk tier's allocation gate and throughput sweep (standard library only).

  ./tools/glm53/build-sycl.sh
  python3 tools/glm53/disk_test.py gate  [--requests 1200] [--fault-addrs 20]
  python3 tools/glm53/disk_test.py bench [--pieces 256,512,1024,4096] [--threads 4,8,16] [--iters 200]
  python3 tools/glm53/disk_test.py clean            # deletes the synthetic store

Store: by default a synthetic K3XS v2 store (512 KiB stride) in $W/tmp/disk_test (W = ~/glm53-work), built once:
layers 3..9 x 256 experts, every record a copy of expert e % 8 of the UD-Q2_K_XL layer 3 slice in $W/data (IQ2_XS /
IQ2_XS / IQ3_XXS, 12.1 MB per record, GLM-5.3's size), 22.5 GB, so the LRU (--lru-gb, 2) cannot hold it. Its rows are
checked against `g53x <slice> check` (x = the check's x). --store PATH (a path inside the container, e.g. a real GLM
store mounted with DSYCL_ARGS="-v /nvme:/store:ro") uses an existing store instead: rows are then not verified.

The server (in the oneAPI container, dsycl.sh): `g53x SET STORE - - serveidx PORT`, SET = experts 0..--resident-1 of
every layer (resident), K3X_DISK_TIER=1 with STORE as the disk store (every other expert, from disk).

gate: the server runs with LD_PRELOAD=allocgate.so (allocgate.c). A warm-up (300 requests, the same mix, more cold
reads than the LRU has slots so evictions happen) on a first connection, 50 more requests on a second, then SIGUSR2
opens the counting window: --requests requests of k = 1..8 rows, each row resident, an LRU hit (one of the layer's last
8 cold reads), a cold read (the layer's next expert, not read for > 1700 reads), or now and then an unservable id (a NaN
row); SIGUSR2 closes it (connection setup and teardown stay outside). Asserted over the window:
zero malloc / calloc / realloc / memalign family / operator new / free / operator delete / mmap / munmap, zero
zeMemAllocHost / Device / Shared / zeMemFree / zeEventCreate / zeEventPoolCreate / zeCommandListCreate(Immediate)
(counted through the loader's exports and DDI tables, see allocgate.c), zero minor and major page faults of the server
(/proc/PID/stat fields 10 and 12, all threads), every row as expected, and the server's disk stats agreeing with the mix
(cold rows = disk reads, LRU-hit rows = LRU hits, no deadline misses). On faults it prints the faulting addresses and
their /proc/PID/maps lines (allocgate's perf_event_open sampler; the container then runs with seccomp unconfined).

bench: per (piece size, readers) a server; per k in 1, 4, 8 and cold fraction 0, 50, 100 % (the other rows resident)
--iters requests on a fresh connection (after 10 warm-up requests), one layer per request: median and p99 latency
(client send to last reply byte), and from the server's disk stats line the drive's GB/s while a read was in flight
(busy) and bytes over the cell's wall time. Deadline --deadline-ms (1000: measure, do not drop rows); rows that missed
it (NaN) are counted.
"""
import argparse
import json
import math
import os
import random
import re
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time

TYPE_IDS = {"Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ4_XS": 23}
HEADER = struct.Struct("<4sIIIIIQQ24x")
ENTRY = struct.Struct("<IIIIQQQQQ8x")
ENTRY_V2 = struct.Struct("<IIIIQQQQQQ")
ALIGN = 4096
STRIDE_ALIGN = 524288
QUANT, SRC_LAYER, SRC_NE = "UD-Q2_K_XL", 3, 8
SYN_LAYERS, SYN_NE = list(range(3, 10)), 256
D = 6144
GATE_KEYS = ["malloc", "calloc", "realloc", "memalign", "new", "free", "mmap", "munmap", "ze_host", "ze_device",
             "ze_shared", "ze_free", "ze_event", "ze_eventpool", "ze_cmdlist"]


def round_up(x, a):
    return (x + a - 1) // a * a


# ---------------------------------------------------------------- host: synthetic store
def build_synthetic(W, path):
    data = os.path.join(W, "data")
    src = {e["proj"]: e for e in json.load(open(os.path.join(data, QUANT + ".json"))) if e["layer"] == SRC_LAYER}
    sizes = [src[w]["bytes_per_expert"] for w in ("gate", "up", "down")]
    types = [TYPE_IDS[src[w]["type"]] for w in ("gate", "up", "down")]
    record = round_up(sum(sizes), ALIGN)
    stride = round_up(record, STRIDE_ALIGN)
    records_offset = round_up(ALIGN + len(SYN_LAYERS) * ENTRY.size, STRIDE_ALIGN)
    total = records_offset + len(SYN_LAYERS) * SYN_NE * stride
    if os.path.exists(path) and os.path.getsize(path) == total and os.path.exists(path + ".json"):
        return
    blobs = [open(os.path.join(data, src[w]["file"]), "rb").read() for w in ("gate", "up", "down")]
    recs = [b"".join(b[e * n:(e + 1) * n] for b, n in zip(blobs, sizes)).ljust(stride, b"\0") for e in range(SRC_NE)]
    t0 = time.time()
    layers = []
    with open(path + ".part", "wb") as f:
        f.write(HEADER.pack(b"K3XS", 2, len(SYN_LAYERS), SYN_NE, STRIDE_ALIGN, ENTRY.size, ALIGN, records_offset))
        for i, L in enumerate(SYN_LAYERS):
            base = records_offset + i * SYN_NE * stride
            layers.append({"layer": L, "base_offset": base})
            f.seek(ALIGN + i * ENTRY.size)
            f.write(ENTRY_V2.pack(L, *types, *sizes, record, base, stride))
        f.seek(records_offset)
        for i in range(len(SYN_LAYERS) * SYN_NE):
            f.write(recs[i % SYN_NE % SRC_NE])
            if i % 64 == 63:   # write back and drop as we go: do not fill the page cache with 22 GB
                f.flush()
                os.fsync(f.fileno())
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                print(f"\r  {f.tell() / 1e9:.1f} / {total / 1e9:.1f} GB", end="", flush=True)
        f.flush()
        os.fsync(f.fileno())
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    dims = {"gate": [D, src["gate"]["ne1"], SYN_NE], "up": [D, src["up"]["ne1"], SYN_NE], "down": [src["down"]["ne0"], D, SYN_NE]}
    with open(path + ".json", "w") as f:
        json.dump({"n_expert": SYN_NE, "records_offset": records_offset, "total_bytes": total, "layers": [
            {"layer": L["layer"], "n_expert": SYN_NE,
             **{w: {"shard": "", "dims": dims[w], "type": types[i], "offset": 0, "n_bytes": sizes[i] * SYN_NE}
                for i, w in enumerate(("gate", "up", "down"))},
             "gate_size": sizes[0], "up_size": sizes[1], "down_size": sizes[2], "record_size": record,
             "base_offset": L["base_offset"], "stride": stride} for L in layers]}, f, indent=1)
    os.rename(path + ".part", path)
    print(f"\r  synthetic store {path}: {total / 1e9:.1f} GB in {time.time() - t0:.0f} s", flush=True)


def host(args):
    T = os.path.dirname(os.path.abspath(__file__))
    W = os.path.expanduser(os.environ.get("W", "~/glm53-work"))
    out = os.path.join(W, "tmp", "disk_test")
    syn = os.path.join(out, "synthetic.k3xs")
    if args.mode == "clean":
        for p in (syn, syn + ".json", syn + ".part"):
            if os.path.exists(p):
                os.remove(p)
        return 0
    os.makedirs(out, exist_ok=True)
    if not args.store:
        build_synthetic(W, syn)
    passthru = [a for a in sys.argv[1:] if a != args.mode]
    cmd = f"python3 /w/tools/disk_test.py inner-{args.mode} " + " ".join(f"'{a}'" for a in passthru)
    env = dict(os.environ)
    if args.mode == "gate" and args.fault_addrs:   # perf_event_open (allocgate's fault sampler) is off in docker's seccomp profile
        env["DSYCL_ARGS"] = (env.get("DSYCL_ARGS", "") + " --security-opt seccomp=unconfined").strip()
    return subprocess.run([os.path.join(T, "dsycl.sh"), cmd], env=env).returncode


# ---------------------------------------------------------------- inner: server + client
def store_layers(path):
    """-> (experts per layer, [layer ids]) from a K3XS header"""
    with open(path, "rb") as f:
        magic, version, nl, ne, sa, esz, align, roff = HEADER.unpack(f.read(HEADER.size))
        assert magic == b"K3XS", path
        out = []
        for i in range(nl):
            f.seek(align + i * esz)
            out.append(struct.unpack("<I", f.read(4))[0])
    return ne, out


def recv_all(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError(f"closed after {len(b)} of {n} bytes")
        b += c
    return bytes(b)


class Server:
    def __init__(self, args, setp, env_extra, preload=False, label="srv"):
        self.port = free_port()
        self.log = f"/w/tmp/disk_test/{label}.log"
        self.setout = f"/w/tmp/disk_test/{label}.setout"
        env = dict(os.environ, K3X_SETOUT=self.setout, K3X_DISK_TIER="1", K3X_DISK_LRU_GB=str(args.lru_gb),
                   K3X_DISK_DEADLINE_MS=str(args.deadline_ms), **env_extra)
        if preload:
            env["LD_PRELOAD"] = "/w/allocgate.so"
        self.lf = open(self.log, "w")
        self.p = subprocess.Popen(["/w/g53x", setp, args.store, "-", "-", "serveidx", str(self.port)],
                                  stderr=self.lf, stdout=self.lf, env=env)
        self.wait_log("serving on", 600)
        for line in open(self.log).read().splitlines():
            if line.startswith(("loaded", "K3X_DISK_TIER", "disk tier: mlock", "self-check")):
                print(f"  [{label}] {line}", flush=True)

    def wait_log(self, what, timeout=30, count=1):
        t0 = time.time()
        while open(self.log).read().count(what) < count:
            if self.p.poll() is not None or time.time() - t0 > timeout:
                raise SystemExit(f"server: no '{what}':\n{open(self.log).read()[-4000:]}")
            time.sleep(0.05)

    def faults(self):
        f = open(f"/proc/{self.p.pid}/stat").read().rsplit(")", 1)[1].split()
        return int(f[7]), int(f[9])   # fields 10 (minflt) and 12 (majflt)

    def thread_faults(self):
        """-> {thread id (name): (minflt, majflt)}"""
        out = {}
        for t in os.listdir(f"/proc/{self.p.pid}/task"):
            try:
                st = open(f"/proc/{self.p.pid}/task/{t}/stat").read()
            except OSError:
                continue
            name = st[st.index("(") + 1:st.rindex(")")]
            f = st.rsplit(")", 1)[1].split()
            out[f"{t} ({name})"] = (int(f[7]), int(f[9]))
        return out

    def disk_stats(self):
        """-> the last "disk:" stats line (written when a connection closes) as a dict"""
        lines = [l for l in open(self.log).read().splitlines() if l.startswith("disk: ")]
        return {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", lines[-1])} if lines else {}

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(10)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.lf.close()


class Client:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port))
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def request(self, gids, x):
        hdr = [len(gids)] + list(gids) + [0] * (16 - len(gids))
        self.s.sendall(struct.pack("<17i", *hdr) + x)
        return recv_all(self.s, len(gids) * D * 4)

    def close(self, port_srv=None):
        self.s.sendall(struct.pack("<17i", *([0] * 17)) + b"\0" * (D * 4))
        try:
            self.s.recv(1)
        except OSError:
            pass
        self.s.close()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def is_nan_row(b):
    v = struct.unpack_from("<4f", b)
    return all(math.isnan(a) for a in v) and all(math.isnan(a) for a in struct.unpack(f"<{D}f", b))


class Workload:
    """the store's servable gids by layer (SETOUT), residents (e < --resident), cold cycles and recent cold reads"""
    def __init__(self, args, srv, NX):
        by = {}
        for l in open(srv.setout):
            L, e = map(int, l.split())
            by.setdefault(L, []).append(e)
        self.NX = NX
        self.layers = sorted(L for L in by if len([e for e in by[L] if e >= args.resident]) >= 32)
        if not self.layers:
            raise SystemExit("no layer with 32 disk experts in SETOUT")
        self.res = {L: [e for e in by[L] if e < args.resident] for L in self.layers}
        self.cold = {L: [e for e in by[L] if e >= args.resident] for L in self.layers}
        self.ptr = {L: 0 for L in self.layers}
        self.recent = {L: [] for L in self.layers}
        self.li = 0
        self.unserved = next(g for g in range(NX * 200) if g // NX not in by)   # a layer not in the store

    def next_layer(self):
        L = self.layers[self.li % len(self.layers)]
        self.li += 1
        return L

    def take_cold(self, L):
        c = self.cold[L]
        e = c[self.ptr[L] % len(c)]
        self.ptr[L] += 1
        self.recent[L] = (self.recent[L] + [e])[-8:]
        return e


def reference(args):
    """-> (x, {source expert: check row}) for the synthetic store, (x, None) otherwise"""
    if args.store != "/w/tmp/disk_test/synthetic.k3xs":
        rnd = random.Random(1)
        return struct.pack(f"<{D}f", *[rnd.uniform(-0.05, 0.05) for _ in range(D)]), None
    sl = {e["proj"]: e for e in json.load(open(f"/w/data/{QUANT}.json")) if e["layer"] == SRC_LAYER}
    ty = [str(TYPE_IDS[sl[w]["type"]]) for w in ("gate", "up", "down")]
    dump = "/w/tmp/disk_test/ref"
    p = subprocess.run(["/w/g53x", f"data/{QUANT}_L{SRC_LAYER}", *ty, "check", dump], capture_output=True, text=True, cwd="/w")
    if p.returncode:
        raise SystemExit(f"check failed:\n{p.stderr}")
    x = open(dump + ".x", "rb").read()
    dn = open(dump + ".dn", "rb").read()
    return x, {e: dn[e * D * 4:(e + 1) * D * 4] for e in range(SRC_NE)}


def write_set(args, path):
    NX, layers = store_layers(args.store)
    with open(path, "w") as f:
        for L in layers:
            for e in range(min(args.resident, NX)):
                f.write(f"{L} {e}\n")
    return NX


def inner_gate(args):
    setp = "/w/tmp/disk_test/gate.set"
    NX = write_set(args, setp)
    x, ref = reference(args)
    srv = Server(args, setp, {"ALLOCGATE_FAULTS": str(args.fault_addrs)} if args.fault_addrs else {}, preload=True, label="gate")
    fails = []
    try:
        wl = Workload(args, srv, NX)
        rnd = random.Random(7)

        def mix_request():
            """-> (gids, kinds): k = 1..8 rows of one layer, each resident / LRU hit / cold, sometimes a NaN id"""
            L = wl.next_layer()
            k = rnd.randint(1, 8)
            gids, kinds, used = [], [], set()
            for _ in range(k):
                what = rnd.choice(("res", "lru", "cold"))
                if what == "lru":
                    cand = [e for e in wl.recent[L] if e not in used]
                    if not cand:
                        what = "cold"
                if what == "res":
                    cand = [e for e in wl.res[L] if e not in used]
                    if not cand:
                        what = "cold"
                e = rnd.choice(cand) if what in ("res", "lru") else wl.take_cold(L)
                used.add(e)
                gids.append(L * NX + e)
                kinds.append(what)
            if rnd.random() < 0.1 and k < 16:
                gids.insert(rnd.randrange(len(gids) + 1), wl.unserved)
                kinds.insert(gids.index(wl.unserved), "nan")
            return gids, kinds

        def run(c, n, count):
            bad = 0
            for _ in range(n):
                gids, kinds = mix_request()
                r = c.request(gids, x)
                for i, (g, kd) in enumerate(zip(gids, kinds)):
                    row = r[i * D * 4:(i + 1) * D * 4]
                    count[kd] = count.get(kd, 0) + 1
                    if kd == "nan" or ref is None:
                        ok = is_nan_row(row) if kd == "nan" else not is_nan_row(row)
                    else:
                        ok = row == ref[(g % NX) % SRC_NE]
                    if not ok and bad < 3:
                        print(f"  bad row: request {gids} kinds {kinds} row {i} ({kd}): "
                              f"{struct.unpack_from('<4f', row)}", flush=True)
                    bad += not ok
            return bad

        # accept/close (and the stats line printed at close) stay outside the window: warm up on one connection, then
        # 50 more requests on the second before the window opens (counted in the stats comparison)
        c = Client(srv.port)
        warm = {}
        bad = run(c, 300, warm)
        c.close()
        srv.wait_log("disk: ", 10, 1)
        st0 = srv.disk_stats()
        c = Client(srv.port)
        pre = {}
        bad += run(c, 50, pre)
        print(f"  warm-up: 300 requests, rows {warm}, disk {st0}", flush=True)
        srv.p.send_signal(signal.SIGUSR2)
        srv.wait_log("ALLOCGATE on", 10)
        f0 = srv.faults()
        t0f = srv.thread_faults()
        cnt = {}
        t0 = time.time()
        bad += run(c, args.requests, cnt)
        dt = time.time() - t0
        f1 = srv.faults()
        t1f = srv.thread_faults()
        srv.p.send_signal(signal.SIGUSR2)
        srv.wait_log("ALLOCGATE window", 10)
        c.close()
        srv.wait_log("disk: ", 10, 2)
        st1 = srv.disk_stats()
        line = [l for l in open(srv.log).read().splitlines() if l.startswith("ALLOCGATE window")][-1]
        win = {k: int(v) for k, v in re.findall(r" (\w+)=(\d+)", line.split("(")[0])}
        dst = {k: st1[k] - st0.get(k, 0) for k in ("hits", "lru_hits", "bytes", "deadline_misses")}
        minflt, majflt = f1[0] - f0[0], f1[1] - f0[1]
        print(f"  window: {args.requests} requests in {dt:.1f} s, rows {cnt}")
        print(f"  {line}")
        print(f"  page faults minor/major: {minflt} {majflt}")
        if minflt or majflt:
            print("    by thread: " + ", ".join(f"{t} {v[0] - t0f.get(t, (0, 0))[0]}/{v[1] - t0f.get(t, (0, 0))[1]}"
                                             for t, v in t1f.items() if v != t0f.get(t, (0, 0))))
        print(f"  disk (window): reads={dst['hits']:.0f} lru_hits={dst['lru_hits']:.0f} bytes={dst['bytes'] / 1e9:.2f} GB "
              f"deadline_misses={dst['deadline_misses']:.0f}; {[l for l in open(srv.log).read().splitlines() if l.startswith('disk: ')][-1]}")
        for k in GATE_KEYS:
            if win.get(k, -1) != 0:
                fails.append(f"{k}={win.get(k)} in the window")
        if minflt or majflt:
            fails.append(f"page faults {minflt}/{majflt}")
            for l in open(srv.log).read().splitlines():
                if l.startswith(("ALLOCGATE fault", "ALLOCGATE faults")):
                    print("    " + l[len("ALLOCGATE "):])
        if bad:
            fails.append(f"{bad} rows not as expected" + ("" if ref else " (NaN where served, or the reverse)"))
        if args.requests < 1000:
            fails.append("fewer than 1000 requests in the window")
        both = {k: cnt.get(k, 0) + pre.get(k, 0) for k in ("cold", "lru")}
        if dst["hits"] != both["cold"] or dst["lru_hits"] != both["lru"] or dst["deadline_misses"]:
            fails.append(f"disk stats {dst} do not match the mix {both} (window and the 50 before it)")
    finally:
        srv.stop()
    if fails:
        print("GATE FAILED: " + "; ".join(fails))
        return 1
    print(f"GATE PASS: {args.requests} requests ({cnt.get('res', 0)} resident, {cnt.get('lru', 0)} LRU-hit, "
          f"{cnt.get('cold', 0)} cold, {cnt.get('nan', 0)} NaN rows" + (", all rows verified" if ref else "") +
          "): zero heap / mmap / Level Zero allocations, zero page faults")
    return 0


def inner_bench(args):
    setp = "/w/tmp/disk_test/bench.set"
    NX = write_set(args, setp)
    x, ref = reference(args)
    results = []
    for piece in map(int, args.pieces.split(",")):
        for thr in map(int, args.threads.split(",")):
            label = f"bench-p{piece}-t{thr}"
            srv = Server(args, setp, {"K3X_DISK_PIECE_KB": str(piece), "K3X_DISK_THREADS": str(thr)}, label=label)
            try:
                wl = Workload(args, srv, NX)
                nst = 0
                prev = {}
                for k in (1, 4, 8):
                    for cold in (0.0, 0.5, 1.0):
                        c = Client(srv.port)
                        ts, nan_rows, bad, ncold = [], 0, 0, 0
                        t_cell = time.perf_counter()
                        for i in range(args.iters + 10):
                            L = wl.next_layer()
                            nc = math.floor((i + 1) * k * cold) - math.floor(i * k * cold)   # cold rows, cumulative
                            es = [wl.take_cold(L) for _ in range(nc)]
                            res = wl.res[L][:]
                            random.Random(i).shuffle(res)
                            es += res[:k - nc]
                            gids = [L * NX + e for e in es]
                            t0 = time.perf_counter_ns()
                            r = c.request(gids, x)
                            t1 = time.perf_counter_ns()
                            if i == 9:
                                t_cell = time.perf_counter()
                            if i >= 10:
                                ts.append((t1 - t0) / 1e3)
                                ncold += nc
                            for j, g in enumerate(gids):
                                row = r[j * D * 4:(j + 1) * D * 4]
                                if is_nan_row(row):
                                    nan_rows += 1
                                elif ref is not None and row != ref[(g % NX) % SRC_NE]:
                                    bad += 1
                        wall = time.perf_counter() - t_cell
                        c.close()
                        nst += 1
                        srv.wait_log("disk: ", 10, nst)
                        st = srv.disk_stats()
                        d = {kk: st[kk] - prev.get(kk, 0) for kk in ("hits", "bytes", "busy_ms", "deadline_misses")}
                        prev = st
                        ts.sort()
                        row = {"piece": piece, "threads": thr, "k": k, "cold": cold, "med": statistics.median(ts),
                               "p99": ts[min(len(ts) - 1, int(math.ceil(0.99 * len(ts))) - 1)],
                               "busy_gbps": d["bytes"] / d["busy_ms"] / 1e6 if d["busy_ms"] else 0.0,
                               "wall_gbps": d["bytes"] / 1e9 / wall if d["bytes"] else 0.0,
                               "reads": d["hits"], "nan": nan_rows, "bad": bad}
                        results.append(row)
                        print(f"  piece {piece:5d} KiB  readers {thr}  k={k}  cold {cold * 100:3.0f} %:  median {row['med']:7.0f} us"
                              f"  p99 {row['p99']:7.0f} us  disk {row['busy_gbps']:5.2f} GB/s busy, {row['wall_gbps']:5.2f} wall"
                              f"  reads {d['hits']:.0f}  NaN rows {nan_rows}  wrong rows {bad}", flush=True)
            finally:
                srv.stop()
    with open("/w/tmp/disk_test/bench.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\n| piece KiB | readers | k | cold % | median us | p99 us | disk GB/s busy | disk GB/s wall |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['piece']} | {r['threads']} | {r['k']} | {r['cold'] * 100:.0f} | {r['med']:.0f} | {r['p99']:.0f} | "
              f"{r['busy_gbps']:.2f} | {r['wall_gbps']:.2f} |")
    wrong = sum(r["bad"] for r in results)
    print(f"\n{len(results)} cells, {sum(r['nan'] for r in results)} NaN rows (deadline {args.deadline_ms} ms), {wrong} wrong rows")
    return 1 if wrong else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["gate", "bench", "clean", "inner-gate", "inner-bench"])
    ap.add_argument("--store", default="", help="store path in the container (default: the synthetic store)")
    ap.add_argument("--resident", type=int, default=8, help="experts 0..N-1 of every layer are resident (SET)")
    ap.add_argument("--lru-gb", type=float, default=2.0)
    ap.add_argument("--deadline-ms", type=float, default=1000.0)
    ap.add_argument("--requests", type=int, default=1200, help="gate: requests in the window")
    ap.add_argument("--pieces", default="256,512,1024,4096", help="bench: K3X_DISK_PIECE_KB values")
    ap.add_argument("--threads", default="4,8,16", help="bench: K3X_DISK_THREADS values (reads in flight)")
    ap.add_argument("--fault-addrs", type=int, default=20,
                    help="gate: print the address and mapping of the first N page faults in the window (0: off)")
    ap.add_argument("--iters", type=int, default=200, help="bench: requests per cell")
    args = ap.parse_args()
    if args.mode.startswith("inner"):
        args.store = args.store or "/w/tmp/disk_test/synthetic.k3xs"
        os.chdir("/w")
        return inner_gate(args) if args.mode == "inner-gate" else inner_bench(args)
    return host(args)


if __name__ == "__main__":
    sys.exit(main())
