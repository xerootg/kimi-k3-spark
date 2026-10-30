"""Compile the real callback with a synthetic GGML adapter; no engine/GPU needed."""
import importlib.util
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

class TraceTest(unittest.TestCase):
    def test_callback_and_check(self):
        with tempfile.TemporaryDirectory() as d:
            binary = Path(d) / 'callback'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            str(Path(__file__).with_name('test_callback.cpp')), '-o', str(binary)], check=True)
            callback_trace = Path(d) / "callback.bin"
            subprocess.run([str(binary), str(callback_trace)], check=True)
            trace = Path(d) / 'trace.bin'
            with trace.open('wb') as f:
                for step in range(2):
                    for layer in range(3, 78):
                        # Scores arrive before/after IDs in alternating steps; no prediction for L3.
                        records = [(3, list(range(8)))]
                        if layer > 3:
                            records.append((1, [float(256-e) for e in range(256)]))
                        for kind, words in records[::1 if step else -1]:
                            f.write(struct.pack('<4i', step, layer, kind, len(words)))
                            f.write(struct.pack('<'+str(len(words))+('f' if kind==1 else 'i'), *words))
            out = subprocess.check_output([str(binary), '--check', str(trace)], text=True)
            self.assertIn('layer 3 real=2 predictor=0 pairs=0 precision=n/a', out)
            self.assertIn('steps=2 layers=75 pairs=148 precision=100.000% (1184/1184)', out)
            spec = importlib.util.spec_from_file_location('place', ROOT/'tools/glm53/place.py')
            place = importlib.util.module_from_spec(spec); spec.loader.exec_module(place)
            actual_complete, actual_preds, actual_discarded = place.load_trace(callback_trace)
            self.assertEqual((len(actual_complete), len(actual_preds), actual_discarded), (2,150,0))
            complete, preds, discarded = place.load_trace(trace)
            self.assertEqual((len(complete), len(preds), discarded), (2,148,0))
            print(out.splitlines()[-1])
            trace.write_bytes(trace.read_bytes()[:-1])
            self.assertNotEqual(subprocess.run([str(binary), '--check', str(trace)], capture_output=True).returncode, 0)

    def test_patch_source_matches(self):
        patch = (ROOT/'patches/0003-expert-store.patch').read_text()
        for name in ('CMakeLists.txt', 'k3-pfdump.cpp', 'trace.h'):
            body = patch.split('+++ b/tools/k3-pfdump/'+name+'\n',1)[1].split('diff --git ',1)[0].splitlines()[1:]
            source = '\n'.join(line[1:] for line in body if line.startswith('+'))+'\n'
            self.assertEqual(source, Path(__file__).with_name(name).read_text())

if __name__ == '__main__':
    unittest.main()
