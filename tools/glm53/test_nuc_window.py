#!/usr/bin/env python3
"""Offline orchestration tests: no SSH, Docker, sockets or model files."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
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
                a.skip_runtime, a.max_gb = True, 18
                a.placement = str(w / 'absent')
                with patch.object(m, 'run') as runtime:
                    m.stage(a, w)
                    runtime.assert_not_called()
                self.assertEqual(len(m.pairs(w / 'resident.set')), 18_000_000_000 // 12_000_000)

    def test_ready_set_keeps_only_resident_owners(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            w = Path(d)
            (w / 'resident.set').write_text('3 9\n3 10\n')
            served = w / 'served.set'
            served.write_text('3 9\n3 10\n3 11\n')
            with contextlib.redirect_stdout(io.StringIO()):
                m.ready_set(types.SimpleNamespace(served=served), w)
            self.assertEqual(m.pairs(w / 'remote.set'), [(3, 9), (3, 10)])
            served.write_text('3 9\n')
            with self.assertRaisesRegex(AssertionError, 'resident set rejected'):
                m.ready_set(types.SimpleNamespace(served=served), w)

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

class ShellTests(unittest.TestCase):
    def shell(self, *args, log_text='loaded 12 of 12 records (1 GB)\ndisk: hits=3'):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            w = Path(d)
            fake = w / 'fake'
            fake.write_text(r"""#!/usr/bin/env python3
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps([name, *args]) + '\n')
if name == 'kubectl':
    if 'logs' in args: print(os.environ['SERVER_LOG'])
    elif 'jsonpath={.spec.replicas}' in args: print('0')
    elif 'rollout' in args: print('successfully rolled out')
if name == 'ssh' and 'pathlib.Path' in args[-1]: print('remote layer 3 rows used=2')
if name == 'docker': sys.exit(99)
""")
            fake.chmod(0o755)
            for name in ('kubectl', 'ssh', 'scp', 'docker'):
                (w / name).symlink_to(fake)
            env = dict(os.environ, PATH=str(w) + ':' + os.environ['PATH'],
                       CALLS=str(w / 'calls'), SERVER_LOG=log_text, G53X=str(w / 'missing'))
            for name in ('DEPLOY', 'NAMESPACE', 'NUC_NODE', 'LEG_LOG'):
                env.pop(name, None)
            result = subprocess.run(['bash', str(Path(__file__).with_name('nuc-window.sh')), *args],
                                    env=env, capture_output=True, text=True)
            calls = [json.loads(line) for line in (w / 'calls').read_text().splitlines()]
            return result, calls

    def test_k8s_stage_skips_binary_and_runtime(self):
        result, calls = self.shell('--k8s', 'stage')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(c[0] == 'docker' for c in calls))
        self.assertTrue(any('scp -q -o BatchMode=yes' in c[-1] and 'placement' in c[-1]
                            for c in calls if c[0] == 'ssh'))
        self.assertTrue(any('--skip-runtime' in c[-1] for c in calls))
        self.assertFalse(any('g53x.new' in c[-1] for c in calls))

    def test_k8s_start_stop_and_missing_count(self):
        result, calls = self.shell('--k8s', 'start')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('served resident records=12', result.stdout)
        self.assertTrue(any('--replicas=1' in c for c in calls))
        self.assertTrue(any('rollout' in c for c in calls))
        self.assertTrue(any('ready-set' in c[-1] for c in calls))
        result, calls = self.shell('--k8s', 'stop')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any('--replicas=0' in c for c in calls))
        result, calls = self.shell('--k8s', 'start', log_text='serving on 19600')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no served count', result.stderr)

    def test_status_reads_spark_leg(self):
        result, calls = self.shell('--k8s', 'status', '/tmp/on1.log')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('disk: hits=3', result.stdout)
        self.assertIn('remote layer 3 rows used=2', result.stdout)
        self.assertTrue(any('pods' in c for c in calls))

if __name__ == '__main__':
    unittest.main(verbosity=2)
