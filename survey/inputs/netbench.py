#!/usr/bin/env python3
"""Tiny TCP bench: bulk throughput (N streams) and request/response latency.

server: netbench.py serve PORT
client: netbench.py bulk HOST PORT SECONDS STREAMS
        netbench.py rr HOST PORT REQ_BYTES RESP_BYTES COUNT
"""
import socket, sys, threading, time, struct

CHUNK = 4 << 20


def recvn(s, n, buf=None):
    mv = memoryview(bytearray(n)) if buf is None else buf[:n]
    got = 0
    while got < n:
        r = s.recv_into(mv[got:], n - got)
        if r == 0:
            raise EOFError
        got += r
    return mv


def handle(c):
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        mode = recvn(c, 1).tobytes()
        if mode == b"B":
            buf = bytearray(CHUNK)
            while c.recv_into(buf):
                pass
        elif mode == b"R":
            req, resp = struct.unpack("!II", recvn(c, 8).tobytes())
            out = b"x" * resp
            rb = memoryview(bytearray(max(req, 1)))
            while True:
                recvn(c, req, rb)
                c.sendall(out)
    except (EOFError, ConnectionError):
        pass
    finally:
        c.close()


def serve(port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(64)
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def bulk(host, port, secs, streams):
    total = [0] * streams
    stop = time.time() + secs

    def run(i):
        s = socket.create_connection((host, port))
        s.sendall(b"B")
        data = b"\0" * CHUNK
        while time.time() < stop:
            s.sendall(data)
            total[i] += CHUNK
        s.close()

    t0 = time.time()
    th = [threading.Thread(target=run, args=(i,)) for i in range(streams)]
    [t.start() for t in th]
    [t.join() for t in th]
    dt = time.time() - t0
    print(f"bulk {host} streams={streams} {sum(total)/dt/1e9*8:.2f} Gbit/s ({sum(total)/dt/1e6:.0f} MB/s)")


def rr(host, port, req, resp, count):
    s = socket.create_connection((host, port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.sendall(b"R" + struct.pack("!II", req, resp))
    out = b"y" * req
    rb = memoryview(bytearray(max(resp, 1)))
    lat = []
    for i in range(count + 20):
        t = time.perf_counter()
        s.sendall(out)
        recvn(s, resp, rb)
        if i >= 20:
            lat.append(time.perf_counter() - t)
    lat.sort()
    p = lambda q: lat[int(q * (len(lat) - 1))] * 1e6
    print(f"rr {host} req={req} resp={resp} n={count} p50={p(.5):.0f}us p90={p(.9):.0f}us p99={p(.99):.0f}us min={p(0):.0f}us")


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "serve":
        serve(int(sys.argv[2]))
    elif m == "bulk":
        bulk(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
    elif m == "rr":
        rr(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
