#!/usr/bin/env python3
"""Offline orchestration tests: no SSH, Docker, sockets or model files."""
import contextlib
import importlib.util
import io
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('window', Path(__file__).with_name('nuc-window.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class WindowTests(unittest.TestCase):
    def test_metadata_budget_and_placement(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            w = Path(d)
            store = w / 'store'
            h = bytearray(64)
            h[:4] = b'K3XS'
            struct.pack_into('<IIII', h, 4, 2, 75, 256, 4096)
            struct.pack_into('<Q', h, 24, 64)
            records = []
            for l in range(3, 78):
                r = bytearray(64)
                struct.pack_into('<I', r, 0, l)
                struct.pack_into('<3Q', r, 16, 4_000_000, 4_000_000, 4_000_000)
                records.append(r)
            store.write_bytes(h + b''.join(records))
            for name in ('rcp', 'ex2p', 'ex2n'):
                (w / (name + '.bin')).touch()
            a = types.SimpleNamespace(placement=str(w / 'absent'))
            with patch.object(m, 'STORE', str(store)), patch.object(m, 'MUFU', str(w)), \
                 patch.object(m, 'run', return_value=types.SimpleNamespace(stdout='libsycl => bundle\n')), \
                 contextlib.redirect_stdout(io.StringIO()):
                m.stage(a, w)
                rows = m.pairs(w / 'resident.set')
                self.assertEqual(len(rows), 44_000_000_000 // 12_000_000)
                self.assertEqual(rows[:2], [(3, 0), (4, 0)])
                self.assertLessEqual(len(rows) * 12_000_000, 44_000_000_000)
                a.placement = str(w / 'placement')
                (w / 'placement').write_text('8 9\n8 10\n8 9\n')
                m.stage(a, w)
                self.assertEqual(m.pairs(w / 'resident.set'), [(8, 9), (8, 10)])

    def test_stale_pid_never_signalled(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            w = Path(d)
            (w / 'nuc.pid.json').write_text('{"pid": 123, "identity": ["old", "/old/g53x"]}')
            with patch.object(m, 'identity', return_value=['new', '/other/g53x']), patch.object(m.os, 'kill') as kill:
                m.terminate(w, 'nuc')
                kill.assert_not_called()
            self.assertFalse((w / 'nuc.pid.json').exists())

    def test_probe_wire_and_fragmentation(self):
        class Connection:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def setsockopt(self, *args): pass
            def sendall(self, data):
                hdr = struct.unpack('<17i', data[:68])
                k = hdr[0]
                assert k in (1, 2)
                assert hdr[1:1+k] == (3*256+9, 3*256+10)[:k]
                assert len(data) == 68 + 6144*4
                self.reply = b'\0' * (k*6144*4)
            def recv(self, n):
                result, self.reply = self.reply[:min(n, 1000)], self.reply[min(n, 1000):]
                return result
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            w = Path(d)
            (w / 'remote.set').write_text('3 9\n3 10\n')
            output = io.StringIO()
            with patch.object(m.socket, 'create_connection', return_value=Connection()), contextlib.redirect_stdout(output):
                m.probe(types.SimpleNamespace(ip='unused', port=1), w)
            self.assertIn('k=1 n=200 p50=', output.getvalue())
            self.assertIn('k=2 n=200 p50=', output.getvalue())

    def test_unresolved_runtime_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            with patch.object(m, 'run', return_value=types.SimpleNamespace(stdout='libur_loader.so => not found\n')), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'unresolved runtime'):
                m.stage(types.SimpleNamespace(), Path(d))

if __name__ == '__main__':
    unittest.main(verbosity=2)
