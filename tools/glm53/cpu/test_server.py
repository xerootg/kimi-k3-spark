#!/usr/bin/env python3
"""Offline CPU wire/disk/allocation regressions. All artifacts stay in this checkout."""

import contextlib
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import math

ROOT = Path(__file__).resolve().parents[3]
CPU = ROOT / "tools/glm53/cpu"
B = CPU / "build/server-test"
D, F = 6144, 2048
PIPE = False
try:
    with socket.socket():
        pass
except PermissionError:
    PIPE = True
    print(
        "Local sockets forbidden: testing identical wire bytes over pipes; TCP integration remains for CI",
        flush=True,
    )
TYPES = [(17, 17, 18), (18, 18, 23), (17, 17, 23), (10, 10, 11), (17, 18, 23)]
SIZES = {17: 74, 18: 98, 23: 136}


def run(args, **kw):
    subprocess.run(list(map(str, args)), check=True, cwd=ROOT, **kw)


def build():
    B.mkdir(parents=True, exist_ok=True)
    (B / "tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(B / "tmp")
    flags = [
        "g++",
        "-O3",
        "-std=c++17",
        "-pthread",
        "-march=x86-64-v3",
        "-ffp-contract=off",
        "-fno-fast-math",
        "-Itools/bitexact",
    ]
    if "avx_vnni" in Path("/proc/cpuinfo").read_text():
        flags += ["-mavxvnni"]
    run(flags + ["-c", CPU / "g53cpu.cpp", "-o", B / "cpu.o"])
    wrap = [
        "-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=posix_memalign,--wrap=aligned_alloc"
    ]
    for name, extra in [
        ("server", []),
        (
            "closed",
            [
                CPU / "test_closed_mufu.cpp",
                "-Wl,--wrap=_Z16g53cpu_load_mufuPKcP11mufu_tables",
            ],
        ),
    ]:
        run(
            flags
            + [
                "-DG53_ALLOC_TEST",
                *(["-DG53_PIPE_TEST"] if PIPE else []),
                CPU / "g53cs.cpp",
                B / "cpu.o",
                *extra,
                *wrap,
                "-o",
                B / name,
            ]
        )
    run(
        flags
        + ["-Itools/glm53/ref", ROOT / "tools/glm53/g53ref_exact.cpp", "-o", B / "ref"]
    )
    # Shared metadata/cache suite already covers pins, eviction, deadlines, failed reads, concurrency and allocation gates.
    run(
        flags
        + [
            ROOT / "tools/bitexact/cpu/test_expert_disk.cpp",
            "-Wl,--wrap=pread",
            *wrap,
            "-o",
            B / "disk-unit",
        ]
    )
    run([B / "disk-unit", B / "unit.k3xs"])


def fixture(real, version=1):
    E = 8
    offset = 4096
    descriptors = []
    idx = []
    with (B / "store").open("wb") as f:
        f.truncate(4096)
        for layer, types in enumerate(TYPES):
            sizes = (
                [
                    F * 24 * SIZES[types[0]],
                    F * 24 * SIZES[types[1]],
                    D * 8 * SIZES[types[2]],
                ]
                if layer != 3
                else [4096] * 3
            )
            stride = (sum(sizes) + 4095) // 4096 * 4096
            descriptors.append(
                struct.pack(
                    "<4I6Q",
                    layer,
                    *types,
                    *sizes,
                    stride,
                    offset,
                    stride if version == 2 else 0,
                )
            )
            records = []
            for e in range(E):
                chunks = []
                for proj, (ty, sz) in enumerate(zip(types, sizes)):
                    if real and layer != 3:
                        src_layer = (
                            8
                            if ty == 18 and proj < 2
                            else 3 if ty == 17 or ty == 18 else 75
                        )
                        path = (
                            Path.home()
                            / f'glm53-work/data/UD-Q2_K_XL_L{src_layer}_{["gate","up","down"][proj]}.bin'
                        )
                        with path.open("rb") as inp:
                            inp.seek(e * sz)
                            chunks.append(inp.read(sz))
                    else:
                        # Nontrivial indexes, signs, scales; zero half weight scale keeps GLU in closed-form subset.
                        bb = SIZES.get(ty, 0)
                        chunks.append(
                            (
                                bytes(2)
                                + bytes((j * 17 + e) & 255 for j in range(bb - 2))
                            )
                            * (sz // bb)
                            if bb
                            else bytes(sz)
                        )
                raw = b"".join(chunks)
                assert len(raw) == sum(sizes)
                raw += bytes(stride - len(raw))
                f.seek(offset)
                f.write(raw)
                records.append(raw)
                idx.append(f"{layer*E+e} {types[0]} {types[2]} {offset}\n")
                offset += stride
            (B / f"L{layer}.records").write_bytes(b"".join(records))
        f.seek(0)
        f.write(
            struct.pack(
                "<4s5I2Q24x", b"K3XS", version, len(TYPES), E, 4096, 64, 64, 4096
            )
        )
        f.write(b"".join(descriptors))
    (B / "all.idx").write_text("".join(idx))
    (B / "ram.idx").write_text(idx[0])


@contextlib.contextmanager
def server(real, idx="all.idx", **env):
    port = 0
    if not PIPE:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("K3X_", "G53"))}
    clean.update(
        K3X_CPU_CPUS=",".join(map(str, sorted(os.sched_getaffinity(0))[:4])),
        K3X_RX_SPIN_US="0",
        K3X_MUFU=str(Path.home() / "glm53-work/mufu"),
        G53_ALLOW_UNLOCKED="1",
    )
    clean.update({k: str(v) for k, v in env.items()})
    with (B / "server.log").open("w+") as log:
        p = subprocess.Popen(
            [
                str(B / ("server" if real else "closed")),
                str(B / idx),
                str(B / "store"),
                str(port),
            ],
            env=clean,
            stderr=log,
            stdin=subprocess.PIPE if PIPE else None,
            stdout=subprocess.PIPE if PIPE else None,
        )
        try:
            end = time.monotonic() + 90
            while True:
                if p.poll() is not None:
                    raise AssertionError((B / "server.log").read_text())
                if PIPE:
                    if "serving on pipes" in (B / "server.log").read_text():
                        break
                    time.sleep(0.02)
                    continue
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    assert time.monotonic() < end, "startup timeout"
                    time.sleep(0.02)
            yield p if PIPE else port
        finally:
            p.terminate()
            p.wait(timeout=10)
    log = (B / "server.log").read_text()
    assert "FAIL request allocations" not in log, log
    assert "PASS request allocations=0" in log, log


def request(port, gids, x):
    if PIPE:
        port.stdin.write(
            struct.pack("<17i", len(gids), *gids, *([0] * (16 - len(gids)))) + x
        )
        port.stdin.flush()
        out = bytearray()
        while len(out) < len(gids) * D * 4:
            q = port.stdout.read(len(gids) * D * 4 - len(out))
            assert q, "short pipe reply"
            out.extend(q)
        return [bytes(out[i * D * 4 : (i + 1) * D * 4]) for i in range(len(gids))]
    with socket.create_connection(("127.0.0.1", port), timeout=90) as s:
        payload = struct.pack("<17i", len(gids), *gids, *([0] * (16 - len(gids)))) + x
        # Fragment the header and payload to exercise partial receives.
        s.sendall(payload[:13])
        s.sendall(payload[13:])
        out = b""
        while len(out) < len(gids) * D * 4:
            q = s.recv(len(gids) * D * 4 - len(out))
            assert q, "short reply"
            out += q
    return [out[i * D * 4 : (i + 1) * D * 4] for i in range(len(gids))]


def nan(row):
    assert all(math.isnan(v) for v in struct.unpack(f"<{D}f", row))


def integration(real):
    x = (
        struct.pack(f"<{D}f", *([127.0] * D))
        if not real
        else struct.pack(f"<{D}f", *[0.001 * ((i * 17) % 1001 - 500) for i in range(D)])
    )
    oracle = {}
    if real:
        inp = B / "input"
        inp.mkdir(exist_ok=True)
        (inp / "x.bin").write_bytes(x)
        (inp / "ids.bin").write_bytes(struct.pack("<8i", *range(8)))
        for layer, types in enumerate(TYPES):
            if layer == 3:
                continue
            out = B / f"expected{layer}"
            run(
                [
                    B / "ref",
                    "records",
                    Path.home() / "glm53-work/mufu",
                    B / f"L{layer}.records",
                    8,
                    *types,
                    inp,
                    out,
                ]
            )
            raw = (out / "dn.bin").read_bytes()
            for e in range(8):
                oracle[layer * 8 + e] = raw[e * D * 4 : (e + 1) * D * 4]
    else:
        oracle = {g: bytes(D * 4) for g in range(40) if g // 8 != 3}
    with server(real) as port:
        for layer in [0, 1, 2, 4]:
            for k in [1, 4, 8, 16]:
                gids = [layer * 8 + (i * 3) % 8 for i in range(k)]
                assert request(port, gids, x) == [oracle[g] for g in gids]
        for row in request(port, [0, 999], x):
            nan(row)
        for row in request(port, [24], x):
            nan(row)
        assert request(port, [0], bytes(D * 4)) == [
            bytes(D * 4)
        ], "valid zero row changed"
    stride = max((B / f"L{i}.records").stat().st_size // 8 for i in range(5))
    with server(
        real,
        "ram.idx",
        K3X_DISK_TIER=1,
        K3X_DISK_LRU_GB=(2 * stride + 1) / 1e9,
        K3X_DISK_DEADLINE_MS=5000,
    ) as port:
        for gids in [[0, 8, 8, 999], [16, 0, 17], [32, 1, 32], [0, 24]]:
            rows = request(port, gids, x)
            for gid, row in zip(gids, rows):
                if gid in oracle:
                    assert row == oracle[gid], f"disk gid {gid}"
                else:
                    nan(row)
    with server(
        real,
        "ram.idx",
        K3X_DISK_TIER=1,
        K3X_DISK_LRU_GB=(stride + 1) / 1e9,
        K3X_DISK_DEADLINE_MS=0,
    ) as port:
        rows = request(port, [8, 0, 999], x)
        nan(rows[0])
        assert rows[1] == oracle[0]
        nan(rows[2])
    print(
        "PASS g53cs: reference wire rows k=1/4/8/16, valid zeros, unsupported/missing ids, RAM/disk, duplicates, partial misses, deadline, zero request allocations",
        flush=True,
    )


def main():
    build()
    real = (
        not os.getenv("G53_TEST_CLOSED")
        and (Path.home() / "glm53-work/data/UD-Q2_K_XL_L3_gate.bin").exists()
        and (Path.home() / "glm53-work/mufu/ex2n.bin").exists()
    )
    if not real:
        print(
            "SKIP table-dependent server rows: slices/MUFU absent; testing closed-form zero-gate fixtures",
            flush=True,
        )
    for version in [1, 2]:
        fixture(real, version)
        integration(real)
    print("PASS ALL GLM CPU server/disk regressions", flush=True)


if __name__ == "__main__":
    main()
