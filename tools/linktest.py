#!/usr/bin/env python3
"""Link test between two hosts (2026-11-09): TCP throughput and small-message round trip.
server: linktest.py server PORT          (serves both tests, exits after 10 min)
client: linktest.py client HOST PORT     (prints one JSON line)
Round trip: 64-byte request, 64-byte reply, TCP_NODELAY, 20000 iterations after 2000 warm-up; also 6 KiB replies
(about one expert row's activations). Throughput: 4 parallel streams for 5 s, 1 MiB writes."""
import json, os, socket, sys, threading, time
SPIN = os.environ.get("SPIN") == "1"

def rt_server(conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        hdr = b""
        while len(hdr) < 8:
            try:
                d = conn.recv(8 - len(hdr), socket.MSG_DONTWAIT) if SPIN else conn.recv(8 - len(hdr))
            except BlockingIOError:
                continue
            if not d:
                return
            hdr += d
        if len(hdr) < 8:
            return
        want = int.from_bytes(hdr[4:], "little")
        n = int.from_bytes(hdr[:4], "little") - 8
        if n > 0:
            conn.recv(n, socket.MSG_WAITALL)
        conn.sendall(b"x" * want)

def sink(conn):
    while conn.recv(1 << 20):
        pass

def server(port):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port)); s.listen(16); s.settimeout(600)
    while True:
        try:
            c, _ = s.accept()
        except socket.timeout:
            return
        kind = c.recv(1, socket.MSG_WAITALL)
        threading.Thread(target=rt_server if kind == b"r" else sink, args=(c,), daemon=True).start()

def roundtrip(host, port, req, rep, iters=20000, warm=2000):
    c = socket.create_connection((host, port)); c.sendall(b"r")
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    msg = (req).to_bytes(4, "little") + rep.to_bytes(4, "little") + b"y" * (req - 8)
    ts = []
    for i in range(iters + warm):
        t = time.perf_counter_ns(); c.sendall(msg)
        got = 0
        while got < rep:
            try:
                got += len(c.recv(rep - got, socket.MSG_DONTWAIT) if SPIN else c.recv(rep - got))
            except BlockingIOError:
                pass
        if i >= warm:
            ts.append(time.perf_counter_ns() - t)
    c.close(); ts.sort()
    return {"p50_us": round(ts[len(ts) // 2] / 1e3, 1), "p99_us": round(ts[int(len(ts) * 0.99)] / 1e3, 1),
            "min_us": round(ts[0] / 1e3, 1)}

def throughput(host, port, streams=4, secs=5.0):
    tot = [0] * streams; buf = b"z" * (1 << 20)
    def f(i):
        c = socket.create_connection((host, port)); c.sendall(b"s"); end = time.time() + secs
        while time.time() < end:
            c.sendall(buf); tot[i] += len(buf)
        c.close()
    th = [threading.Thread(target=f, args=(i,)) for i in range(streams)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
    return round(sum(tot) * 8 / (time.time() - t0) / 1e9, 2)

if __name__ == "__main__":
    if sys.argv[1] == "server":
        server(int(sys.argv[2]))
    else:
        h, p = sys.argv[2], int(sys.argv[3])
        print(json.dumps({"host": h, "rt_64B": roundtrip(h, p, 64, 64), "rt_6KiB": roundtrip(h, p, 64, 6144),
                          "gbps_4streams": throughput(h, p)}), flush=True)
