import hashlib
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "repack-experts.py"
HEADER = struct.Struct("<4sIIIIIQQ24x")
ENTRY_V2 = struct.Struct("<IIIIQQQQQQ")
ALIGN = 4096
TYPE_LAYOUTS = {
    0: (1, 4),
    1: (1, 2),
    8: (32, 34),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 50),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    29: (256, 56),
    30: (1, 2),
}
V1_BASELINE_SHA256 = "1dfab5cc7ca9504b490dceeb81fc22ee8ce4a123cbe50b17d2e4a74c29964d06"


def pack_string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_shard(path, specs, arch):
    metadata = (
        pack_string("general.architecture") + struct.pack("<I", 8) + pack_string(arch)
        + pack_string("general.alignment") + struct.pack("<II", 4, 32)
    )
    infos = bytearray()
    offsets = []
    data = bytearray()
    for spec in specs:
        nelements = 256 * spec["n_expert"]
        block_elems, block_bytes = TYPE_LAYOUTS[spec["type"]]
        payload_size = nelements // block_elems * block_bytes
        offset = (len(data) + 31) // 32 * 32
        data.extend(b"\0" * (offset - len(data)))
        payload = bytes((spec["seed"] + i) % 256 for i in range(payload_size))
        data.extend(payload)
        offsets.append((spec["name"], offset, payload, payload_size // spec["n_expert"]))
        infos.extend(pack_string(spec["name"]))
        infos.extend(struct.pack("<IQQQI", 3, 256, 1, spec["n_expert"], spec["type"]))
        infos.extend(struct.pack("<Q", offset))

    header = b"GGUF" + struct.pack("<IQQ", 3, len(specs), 2) + metadata + infos
    data_start = (len(header) + 31) // 32 * 32
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\0" * (data_start - len(header)))
        f.write(data)
    return {
        name: (path, data_start + offset, payload, expert_size)
        for name, offset, payload, expert_size in offsets
    }


def make_model(directory, arch, layer_types, n_expert, dense_layers=(), split=True):
    specs = []
    for layer, types in layer_types.items():
        if layer in dense_layers:
            continue
        for name, type_id, seed in zip(("gate", "up", "down"), types, (layer * 17, layer * 17 + 5, layer * 17 + 11)):
            specs.append({
                "name": f"blk.{layer}.ffn_{name}_exps.weight",
                "type": type_id,
                "n_expert": n_expert,
                "seed": seed,
            })

    if split:
        first = Path(directory) / "model-00001-of-00002.gguf"
        second = Path(directory) / "model-00002-of-00002.gguf"
        refs = write_shard(first, specs[::2], arch)
        refs.update(write_shard(second, specs[1::2], arch))
    else:
        first = Path(directory) / "model.gguf"
        refs = write_shard(first, specs, arch)
    return str(first), refs


def run_script(gguf, out, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--gguf", gguf, "--out", str(out), *args],
        capture_output=True,
        text=True,
    )


class RepackExpertsTests(unittest.TestCase):
    def test_default_output_matches_v1_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            gguf, _ = make_model(
                directory, "kimi_k3", {0: (12, 12, 8), 1: (8, 12, 12)}, 2
            )
            out = Path(directory) / "store"
            result = run_script(gguf, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            digest = hashlib.sha256(out.read_bytes()).hexdigest()
            self.assertEqual(digest, V1_BASELINE_SHA256)

    def test_v2_layout_records_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            layer_types = {0: (12, 12, 8), 1: (8, 12, 12)}
            gguf, refs = make_model(directory, "kimi_k3", layer_types, 2)
            out = Path(directory) / "store"
            result = run_script(gguf, out, "--stride-align", "8192")
            self.assertEqual(result.returncode, 0, result.stderr)

            data = out.read_bytes()
            magic, version, n_layers, n_expert, align, entry_size, index_offset, records_offset = (
                HEADER.unpack_from(data)
            )
            self.assertEqual((magic, version, n_layers, n_expert, align, entry_size),
                             (b"K3XS", 2, 2, 2, 8192, 64))
            entries = []
            for i in range(n_layers):
                entry = ENTRY_V2.unpack_from(data, index_offset + i * entry_size)
                layer, gate_ty, up_ty, down_ty, gate_size, up_size, down_size, record_size, base, stride = entry
                self.assertEqual(stride, 8192)
                self.assertEqual(record_size, ALIGN)
                self.assertEqual(base % align, 0)
                self.assertEqual((gate_ty, up_ty, down_ty), layer_types[layer])
                entries.append((layer, gate_size, up_size, down_size, record_size, base, stride))

            self.assertEqual(records_offset % align, 0)
            for layer, gate_size, up_size, down_size, record_size, base, stride in entries:
                tensor_names = (
                    f"blk.{layer}.ffn_gate_exps.weight",
                    f"blk.{layer}.ffn_up_exps.weight",
                    f"blk.{layer}.ffn_down_exps.weight",
                )
                payloads = [refs[name][2] for name in tensor_names]
                expert_sizes = [refs[name][3] for name in tensor_names]
                self.assertEqual(expert_sizes, [gate_size, up_size, down_size])
                for expert in range(n_expert):
                    slot = base + expert * stride
                    self.assertEqual(slot % align, 0)
                    expected = b"".join(
                        payload[expert * size:(expert + 1) * size]
                        for payload, size in zip(payloads, expert_sizes)
                    )
                    self.assertEqual(data[slot:slot + len(expected)], expected)
                    self.assertEqual(data[slot + len(expected):slot + record_size],
                                     b"\0" * (record_size - len(expected)))
                    self.assertEqual(data[slot + record_size:slot + stride],
                                     b"\0" * (stride - record_size))

    def test_glm_and_k3_like_indices(self):
        cases = (
            ("kimi_k3", {0: (12, 12, 8), 1: (8, 12, 12), 2: (12, 8, 12)}, 2, ()),
            ("glm-dsa", {0: (12, 12, 12), 1: (12, 12, 12), 2: (17, 17, 18), 3: (17, 18, 18)}, 3, (0, 1)),
        )
        for arch, layers, n_expert, dense in cases:
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                gguf, _ = make_model(directory, arch, layers, n_expert, dense_layers=dense)
                result = run_script(gguf, Path(directory) / "store", "--index-only")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"{len(layers) - len(dense)} MoE layers x {n_expert} experts", result.stdout)
                self.assertIn("gate", result.stdout)

    def test_all_glm_quant_types_are_supported(self):
        types = (19, 29, 16, 17, 22, 18, 21, 23, 10, 11, 12, 13, 14, 8, 30, 1, 0)
        layer_types = {
            layer: tuple(types[layer * 3:layer * 3 + 3])
            for layer in range((len(types) + 2) // 3)
        }
        layer_types[len(layer_types) - 1] = layer_types[len(layer_types) - 1] + (0,) * (
            3 - len(layer_types[len(layer_types) - 1])
        )
        with tempfile.TemporaryDirectory() as directory:
            gguf, _ = make_model(directory, "glm-dsa", layer_types, 2)
            result = run_script(gguf, Path(directory) / "store", "--index-only")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_index_only_reports_larger_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            gguf, _ = make_model(directory, "glm-dsa", {3: (17, 17, 18)}, 2)
            result = run_script(gguf, Path(directory) / "store", "--index-only", "--stride-align", "8192")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("stride 8192 B", result.stdout)

    def test_invalid_stride_alignments_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            gguf, _ = make_model(directory, "kimi_k3", {0: (12, 12, 8)}, 2)
            for value in ("3000", "0", "1048577"):
                with self.subTest(value=value):
                    result = run_script(gguf, Path(directory) / "store", "--stride-align", value)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("power of two and at least 4096 bytes", result.stderr)


if __name__ == "__main__":
    unittest.main()
