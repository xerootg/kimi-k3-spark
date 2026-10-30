#!/usr/bin/env python3
"""CPU-only disk regression suite (Linux x86-64; g++, curl, Python stdlib).

Run from any directory:
  python3 tools/bitexact/cpu/test_disk_tier.py
  python3 tools/bitexact/cpu/test_disk_tier.py --uring  # if liburing is installed
  python3 tools/bitexact/cpu/test_disk_tier.py --baseline-ref <pre-change-commit>

Default baseline: the parent of the first commit adding ../expert_disk.h (HEAD
before that file is committed). Build and fixture files use a unique directory under $TMPDIR (default /tmp; it must be a block
filesystem, not tmpfs, for the O_DIRECT cases),
which is removed even after failures. No production files are changed.
The only download is ggml-common.h at versions.env's LLAMA_SHA, into the
automatically cleaned workspace; repository dependency files are not changed.

Exactness scope: byte-identical RAM/disk/opt-out wire replies through the real
unmodified k3cpu.cpp kernels, with a linker-wrapped synthetic packed MUFU loader
ONLY in test executables. These are storage/serving regressions, not validation
of NVIDIA MUFU accuracy. No production tables/model/GPU needed. Without AVX-VNNI
we preinclude an exact AVX2 integer-dot implementation, leaving source kernels
and FP operations unchanged. Native AVX-VNNI is used automatically if available.
The server serializes connections; parallel clients test that existing contract,
while shared-header tests exercise genuinely concurrent cache readers.
Deterministically delayed pread hooks run only in the non-liburing build;
ordinary deadlines, failed reads, concurrency and real-kernel integration run
in both builds, whether native rings initialize or fall back under restrictions.
"""
import argparse
import concurrent.futures
import contextlib
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import tempfile
import time

CPU = Path(__file__).resolve().parent
ROOT = CPU.parents[2]
WORK = None
D, F = 3584, 3072
ROW = D * 4
MUFU_SYMBOL = "_Z15k3cpu_load_mufuPKcP12k3_mufu_tabs"


def run(args, **kw):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kw)


def baseline_source(ref):
    if ref:
        source = subprocess.check_output(
            ["git", "show", f"{ref}:tools/bitexact/cpu/k3cs.cpp"], cwd=ROOT)
    else:
        added = subprocess.check_output(
            ["git", "log", "--diff-filter=A", "--format=%H", "--",
             "tools/bitexact/expert_disk.h"], cwd=ROOT, text=True).splitlines()
        ref = added[-1] + "^" if added else "HEAD"
        source = subprocess.check_output(
            ["git", "show", f"{ref}:tools/bitexact/cpu/k3cs.cpp"], cwd=ROOT)
    assert b"K3X_DISK_TIER" not in source, "baseline must predate the disk tier"
    target = WORK / "baseline_k3cs.cpp"
    target.write_bytes(source)
    return target


def build(uring, baseline):
    include = WORK / "cpu"
    include.mkdir()
    common = WORK / "sycl" / "ggml-common.h"
    common.parent.mkdir()
    sha = re.search(r"^LLAMA_SHA=([0-9a-f]{40})$",
                    (ROOT / "versions.env").read_text(), re.M).group(1)
    run(["curl", "--fail", "--location", "--retry", "3",
         f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{sha}/ggml/src/ggml-common.h",
         "--output", common])
    cxx = os.environ.get("CXX", "g++")
    production_env = dict(os.environ, OUT=str(WORK / "production-k3cs"),
                          EXTRA=f"-I{include}")
    run(["bash", CPU / "build_k3cs.sh"], env=production_env)
    usage = subprocess.run([str(WORK / "production-k3cs")],
                           capture_output=True, text=True, timeout=5)
    assert usage.returncode == 1 and "usage: k3cs" in usage.stderr
    print("PASS production static build script and executable startup", flush=True)
    flags = [cxx, "-O2", "-g", "-std=c++17", "-pthread", "-I", include, "-I", CPU]
    io = ["-DK3X_HAVE_URING", "-luring"] if uring else []
    run(flags + [CPU / "test_expert_disk.cpp", "-Wl,--wrap=pread",
                 "-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc",
                 "-Wl,--wrap=posix_memalign,--wrap=aligned_alloc",
                 "-o", WORK / "unit"] + io)
    run([WORK / "unit", WORK / "unit.k3xs"], timeout=45)
    flags += ["-ffp-contract=off", "-fno-fast-math", "-march=x86-64-v3"]
    run(flags + [CPU / "test_disk_avx2.cpp", "-o", WORK / "integer-dot"])
    run([WORK / "integer-dot"])
    cpuinfo = Path("/proc/cpuinfo").read_text()
    if re.search(r"\bavx_vnni\b", cpuinfo):
        flags += ["-mavxvnni"]
        print("Native AVX-VNNI kernels", flush=True)
    else:
        flags += ["-include", CPU / "test_disk_avx2.h"]
        print("AVX2 exact integer-dot compatibility (no AVX-VNNI on host)", flush=True)
    run(flags + ["-c", CPU / "k3cpu.cpp", "-o", WORK / "kernels.o"])
    for name, source, extra in [
            ("server", CPU / "k3cs.cpp", io),
            ("baseline", baseline, [])]:
        run(flags + [source, WORK / "kernels.o", CPU / "test_disk_mufu.cpp",
                     f"-Wl,--wrap={MUFU_SYMBOL}", "-o", WORK / name] + extra)


def record(types, expert):
    """GGUF gate|up|down, actual 3584x3072 dimensions, finite half scales."""
    matrices = []
    for matrix, (t, rows, cols) in enumerate(zip(types, [F, F, D], [D, D, F])):
        block_size = 50 if t == 19 else 66
        # Distinct repeated 256-value blocks per row; patterned bytes cover grid
        # indexes, signs and scales. IQ1_S qh is 8 LE u16 after 32 qs bytes.
        blocks = []
        for r in range(32):
            seed = 7 + expert * 23 + matrix * 13 + r * 3
            scale = struct.pack("<H", 0x1800 + expert * 0x100 + r * 3)
            qs = bytes((seed + j * 17) & 255 for j in range(block_size - 2))
            if t == 16 and matrix < 2:
                # Positive gate/up weights avoid the deliberately tiny EX2
                # fixture's overflow boundary making an entire expert zero.
                qs = (bytes((seed + j * 17) & 255 for j in range(4)) + bytes(4)) * 8
            blocks.append((scale + qs) * (cols // 256))
        matrices.append(b"".join(blocks) * (rows // len(blocks)))
    return b"".join(matrices)


def fixture():
    """Documented K3XS v1 fixture, LE, 64B header/index, 4096B alignment.

    Header: magic/version/layer_count/expert_count/alignment/index_entry_size
    at 0/4/8/12/16/20 (u32), index/data offsets at 24/32 (u64).
    Layer entry: layer/tg/tu/td at 0/4/8/12 (u32); gate/up/down bytes,
    aligned record stride, first record offset at 16/24/32/40/48 (u64).
    GID = layer*expert_count+expert. Two real CPU types plus an unsupported
    IQ2_XS layer verify rejection rather than reinterpretation by the server.
    """
    path = WORK / "full.k3xs"
    data_offset = 4096
    descriptors, payload, index, heterogeneous = [], [], [], []
    offset = data_offset
    for layer, types in enumerate([(19, 19, 19), (16, 16, 16),
                                   (17, 17, 17), (19, 16, 19)]):
        lengths = [rows * (cols // 256) * (50 if t == 19 else 66)
                   for t, rows, cols in zip(types, [F, F, D], [D, D, F])]
        if types[0] == 17:
            lengths = [1024] * 3
        stride = (sum(lengths) + 4095) // 4096 * 4096
        descriptors.append(struct.pack("<4I5Q8x", layer, *types,
                                       *lengths, stride, offset))
        for expert in range(2):
            gid = layer * 2 + expert
            raw = record(types, gid) if types[0] != 17 else b"\x5a" * sum(lengths)
            assert len(raw) == sum(lengths)
            payload.append(raw + bytes(stride - len(raw)))
            line = f"{gid} {types[0]} {types[2]} {offset}\n"
            if layer < 2:
                index.append(line)
            elif layer == 3:
                heterogeneous.append(line)
            offset += stride
    header = struct.pack("<4s5I2Q24x", b"K3XS", 1, 4, 2, 4096, 64, 64, data_offset)
    with path.open("wb") as out:
        out.write(header)
        out.write(b"".join(descriptors))
        out.write(bytes(data_offset - out.tell()))
        for raw in payload:
            out.write(raw)
    (WORK / "all.idx").write_text("".join(index))
    (WORK / "mixed.idx").write_text(index[0])
    (WORK / "heterogeneous.idx").write_text("".join(heterogeneous))
    return max(len(p) for p in payload)


@contextlib.contextmanager
def server(binary="server", idx="all.idx", **env):
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(("K3X_", "K3CPU_"))}
    cpus = sorted(os.sched_getaffinity(0))[:2]
    clean.update(K3X_CPU_CPUS=",".join(map(str, cpus)), K3X_CPU_SPIN_US="0",
                 K3X_RX_SPIN_US="0", K3X_MAX_GB="1", K3X_MUFU="test-only",
                 K3X_DISK_THREADS="4", K3X_DISK_DEADLINE_MS="5000")
    clean.update({k: str(v) for k, v in env.items()})
    log_path = WORK / f"{binary}-{port}.log"
    with log_path.open("w") as log:
        p = subprocess.Popen([str(WORK / binary), str(WORK / idx),
                              str(WORK / "full.k3xs"), str(port)],
                             env=clean, stdout=log, stderr=log)
        try:
            end = time.monotonic() + 20
            while time.monotonic() < end:
                if p.poll() is not None:
                    raise AssertionError(log_path.read_text())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        break
                except OSError:
                    time.sleep(.02)
            else:
                raise AssertionError("server did not start: " + log_path.read_text())
            yield port, log_path
            assert p.poll() is None, log_path.read_text()
        except BaseException:
            print(log_path.read_text(), flush=True)
            raise
        finally:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def request(port, gids, seed=0):
    header = struct.pack("<17i", len(gids), *gids, *([0] * (16 - len(gids))))
    x = struct.pack(f"<{D}f", *[.03 * ((i * 53 + seed * 11) % 67 + 1)
                               for i in range(D)])
    with socket.create_connection(("127.0.0.1", port), timeout=15) as s:
        s.sendall(header + x)
        result = bytearray()
        while len(result) < len(gids) * ROW:
            chunk = s.recv(len(gids) * ROW - len(result))
            assert chunk, "short server reply"
            result.extend(chunk)
    return [bytes(result[i:i + ROW]) for i in range(0, len(result), ROW)]


def nan(row):
    assert all(math.isnan(x) for x in struct.unpack(f"<{D}f", row)), "expected all-NaN row"


def finite(row):
    values = struct.unpack(f"<{D}f", row)
    assert all(math.isfinite(x) for x in values), "oracle must not be NaN/inf"
    assert any(x != 0 for x in values), "oracle must not be vacuous zero"


def integration(stride):
    with server(binary="baseline", K3X_SETOUT=WORK / "baseline.set") as (old, _), server(
            K3X_DISK_TIER="0", K3X_DISK_STORE="/does-not-exist",
            K3X_DISK_LRU_GB="-1", K3X_SETOUT=WORK / "optout.set") as (ram, _):
        assert (WORK / "baseline.set").read_bytes() == (WORK / "optout.set").read_bytes()
        oracle = {}
        for seed in range(3):
            rows = request(ram, [0, 1, 2, 3], seed)
            assert rows == request(old, [0, 1, 2, 3], seed), "opt-out changed baseline bytes"
            for gid, row in enumerate(rows):
                finite(row)
                oracle[seed, gid] = row
        assert len(set(oracle.values())) >= 4, "synthetic experts must produce distinct bytes"
        # Legacy opt-out all-or-nothing misses stay byte-identical to HEAD.
        assert request(ram, [0, 999]) == request(old, [0, 999])
        for row in request(ram, [0, 999]):
            nan(row)
    print("PASS opt-out byte-identical to pre-disk HEAD (3 inputs, both CPU types)", flush=True)
    with server() as (unset, _):
        assert request(unset, [0, 1, 2, 3]) == [oracle[0, g] for g in range(4)]
    print("PASS disk tier unset preserves baseline bytes", flush=True)

    budget = f"{(2 * stride + 1) / 1e9:.12f}"
    with server(idx="mixed.idx", K3X_DISK_TIER="1", K3X_DISK_LRU_GB=budget,
                K3X_SETOUT=WORK / "disk.set") as (port, log):
        assert (WORK / "disk.set").read_text() == "0 0\n0 1\n1 0\n1 1\n3 0\n3 1\n", (
            "disk set must include every supported expert, not only the RAM index")
        with server(idx="heterogeneous.idx", K3X_DISK_TIER="1",
                    K3X_DISK_LRU_GB=budget) as (typed_ram, _):
            typed_rows = request(typed_ram, [6, 7])
            for row in typed_rows:
                finite(row)
            assert request(port, [6, 7]) == typed_rows, "separate up type changed RAM/disk bytes"
        for seed in range(3):
            rows = request(port, [0, 1, 2, 999], seed)
            assert rows[:3] == [oracle[seed, gid] for gid in [0, 1, 2]]
            nan(rows[3])
        # Duplicate disk GIDs share one pinned slot; order survives compaction.
        rows = request(port, [2, 0, 2, 999, 1])
        assert [rows[i] for i in [0, 1, 2, 4]] == [oracle[0, g] for g in [2, 0, 2, 1]]
        nan(rows[3])
        for row in request(port, [4, 5]):  # unsupported IQ2_XS types
            nan(row)
        def client(n):
            gids = [0, 1, 2] if n % 2 else [3, 0, 3]
            assert request(port, gids, n % 3) == [oracle[n % 3, g] for g in gids]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(client, range(12)))
        # A dropped client must not unpin weights while compute is active.
        with socket.create_connection(("127.0.0.1", port), timeout=15) as s:
            s.sendall(struct.pack("<17i", 2, 1, 2, *([0] * 14)) +
                      struct.pack(f"<{D}f", *([.25] * D)))
        assert request(port, [2, 1]) == [oracle[0, 2], oracle[0, 1]]
        time.sleep(.05)
        assert "disk: hits=" in log.read_text()
    print("PASS RAM/disk byte equality; separate up type, full .set, partial rows, duplicates, parallel clients", flush=True)

    with server(idx="mixed.idx", K3X_DISK_TIER="1",
                K3X_DISK_LRU_GB=f"{(stride + 1) / 1e9:.12f}") as (port, _):
        rows = request(port, [0, 1, 2, 999])
        assert rows[:2] == [oracle[0, 0], oracle[0, 1]]
        nan(rows[2]); nan(rows[3])
        for gid in [2, 3, 1, 2, 1, 3]:
            assert request(port, [gid])[0] == oracle[0, gid], "LRU eviction corrupted bytes"
    print("PASS one-slot budget, pinned capacity miss and repeated LRU eviction", flush=True)

    for short in ["0", "0.0001"]:
        with server(idx="mixed.idx", K3X_DISK_TIER="1",
                    K3X_DISK_LRU_GB=budget, K3X_DISK_DEADLINE_MS=short) as (port, _):
            rows = request(port, [1, 0, 999])
            nan(rows[0]); assert rows[1] == oracle[0, 0]; nan(rows[2])
            # A completed late read can be used on a later request, never in the
            # expired one; wait deterministically by retrying the cached record.
            end = time.monotonic() + 5
            while True:
                row = request(port, [1])[0]
                if row == oracle[0, 1]:
                    break
                nan(row)
                assert time.monotonic() < end, "late read never became reusable"
                time.sleep(.01)
    print("PASS zero/sub-microsecond deadline, valid RAM partial row and late completion reuse", flush=True)


def main():
    global WORK
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uring", action="store_true")
    parser.add_argument("--baseline-ref")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="k3cs-disk-") as work:
        WORK = Path(work)
        baseline = baseline_source(args.baseline_ref)
        build(args.uring, baseline)
        integration(fixture())
        print("PASS ALL CPU disk-tier regressions" + (" (liburing build)" if args.uring else " (pread)"))


if __name__ == "__main__":
    main()
