#!/usr/bin/env python3
"""Check a running f53s over the wire: replay flashrow cases (T tokens, 8 records of layer L with known expert ids),
send the pairs the box owns in one batched request per case, compare the rows bit for bit, time the round trips."""
import socket, struct, sys, time, array
host, port, frb, layer = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
exps = [int(x) for x in sys.argv[5].split(",")]          # the 8 records' expert ids, in record order
owned = set(int(x) for x in sys.argv[6].split(","))       # experts the box owns
magic = int(sys.argv[7], 16) if len(sys.argv) > 7 else 0x4E440000   # 0x4E450000: rows in the one-warp MoE kernel order
D, KND = 4096, 32
b = open(frb, "rb").read()
nc, T, NE = struct.unpack_from("<3i", b, 0); o = 12
s = socket.create_connection((host, port)); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
def recvn(n):
    buf = bytearray()
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c: raise SystemExit("closed")
        buf += c
    return bytes(buf)
bad = rows = 0; rtts = []
for ci in range(nc):
    x = b[o:o + T*D*4]; o += T*D*4
    ids = struct.unpack_from(f"<{T*NE}i", b, o); o += T*NE*4
    ref = b[o:o + T*NE*D*4]; o += T*NE*D*4
    pairs = [(t, k) for t in range(T) for k in range(NE) if exps[ids[t*NE+k]] in owned][:KND]
    hdr = [magic | len(pairs), T, ci, layer]
    for (t, k) in pairs: hdr += [layer*288 + exps[ids[t*NE+k]], t]
    hdr += [0] * (4 + 2*KND - len(hdr))
    t0 = time.perf_counter()
    s.sendall(struct.pack(f"<{len(hdr)}i", *[h if h < 2**31 else h - 2**32 for h in hdr]) + x)
    rep = recvn(len(pairs)*D*4)
    rtts.append((time.perf_counter() - t0) * 1e3)
    for j, (t, k) in enumerate(pairs):
        rows += 1
        if rep[j*D*4:(j+1)*D*4] != ref[((t*NE)+k)*D*4:((t*NE)+k+1)*D*4]: bad += 1
rtts.sort()
print(f"{frb}: {nc} cases, {rows} rows over the wire, {bad} differ; RTT p50 {rtts[len(rtts)//2]:.2f} ms max {rtts[-1]:.2f} ms ({rows/nc:.1f} pairs per request)")
