"""Offline regression coverage using the captured Spark evidence as fixtures."""
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/glm53/window_digest.py'
FIXTURES = ROOT / 'docs/evidence/glm-windows-2026-10-30'
SPEC = importlib.util.spec_from_file_location('window_digest', SCRIPT)
digest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(digest)


class WindowDigestTests(unittest.TestCase):
    def fixture(self, name):
        return digest.digest((FIXTURES / name).read_text(), name)

    def test_u_means_and_unlabeled_hooks(self):
        row = self.fixture('runner2-u.txt')
        self.assertIn('CB0 r1/r2 4.797/4.696', row)
        self.assertIn('off 4.798, on 5.427 tok/s; gain +13.1 %', row)
        self.assertIn('all 8 observed short-prompt passes', row)
        self.assertIn('unlabeled host bank hooks:', row)
        self.assertIn('real-hook 41.03 ms/token', row)
        self.assertIn('real-hook 45.48 ms/token', row)

    def test_l_final_fill_not_intermediate_or_ubatch(self):
        rows = self.fixture('runner3-s-l.txt').splitlines()
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0].startswith('| s (21:23) |'))
        self.assertIn('build19=0', rows[0])
        self.assertIn('SIGTERM to exit 8.3 s', rows[0])
        row = rows[1]
        self.assertIn('LF0 fill_late: expert store: 237519 lookups, hit rate 92.0%', row)
        self.assertIn('LF1b fill_late: expert store: 237519 lookups, hit rate 92.8%', row)
        self.assertNotIn('35483 lookups', row)
        self.assertIn('hit rate = 68.09%', row)
        self.assertIn('gain +5.5 %', row)
        self.assertIn('build18b=0', row)
        self.assertIn('**verdict**', rows[2])

    def test_w_buckets_and_denominator(self):
        row = self.fixture('legs-w.out')
        self.assertTrue(row.startswith('| w |'))
        self.assertIn('build12c=0 22:03', row)
        self.assertIn('real-hook 53.45 ms/token', row)
        self.assertIn('HW1b host bank real bucket: e_other=28875/514.611ms', row)
        self.assertIn('gain -1.0 %', row)
        self.assertIn('hit rate = 59.17%', row)
        self.assertIn('decode `fill_late ... hit rate`: not in', row)
        self.assertEqual(row.count('|'), 4)

    def test_i_excludes_warmup_and_extra_deadline_from_means(self):
        row = self.fixture('runner4-i.txt')
        off = (5.388 + 5.397 + 5.39 + 5.398) / 4
        on = (5.027 + 4.974 + 5.12 + 4.969) / 4
        means = re.search(r'off ([0-9.]+), on ([0-9.]+)', row)
        self.assertAlmostEqual(float(means[1]), off, delta=0.00051)
        self.assertAlmostEqual(float(means[2]), on, delta=0.00051)
        self.assertIn(f'gain {100 * (on / off - 1):+.1f} %', row)
        self.assertIn('all 15 observed short-prompt passes', row)
        self.assertIn('R1d r1/r2 5.201/5.125', row)
        self.assertIn('GATE-FAIL', row)
        self.assertIn('hit rate = 59.66%', row)

    def test_every_fixture_is_a_three_column_row(self):
        for path in sorted(FIXTURES.iterdir()):
            if not (path.name.startswith('runner') or path.name.startswith('legs-')):
                continue
            with self.subTest(path=path.name):
                rows = self.fixture(path.name).splitlines()
                self.assertTrue(rows)
                for row in rows:
                    self.assertEqual(row.count('|'), 4)
                    self.assertTrue(row.endswith('**verdict** |'))

    def test_truncated_and_timestamped_inputs(self):
        for name in ('runner2-u.txt', 'runner3-s-l.txt', 'legs-w.out', 'runner4-i.txt'):
            source = (FIXTURES / name).read_text()
            clipped = '\n'.join(line[:200] for line in source.splitlines())
            self.assertEqual(digest.digest(clipped, name), digest.digest(source, name))
            stamped = '\n'.join('1.23.456.789 I ' + line for line in clipped.splitlines())
            self.assertEqual(digest.digest(stamped, name), digest.digest(clipped, name))

    def test_long_prompt_hashes_and_pipe_escaping(self):
        row = self.fixture('legs-v.out')
        self.assertIn('L0 decode 4.09 tok/s md5 25d4133dc7', row)
        self.assertIn('all 8 observed short-prompt passes', row)
        rows = self.fixture('runner-p-q-r-s.txt').splitlines()
        self.assertIn('md5 bf747563', rows[1])
        self.assertIn('&#124;', rows[1])
        row = self.fixture('legs-o.txt')
        self.assertIn('de5778fb0a178741a25d68ded6871e99', row)
        self.assertIn('pp=24.452 tok/s', row)

    def test_missing_and_mismatched_hashes_incomplete_quartet(self):
        row = digest.digest('X0 r1 5 tok/s md5 deadbeef\nX0 r2 6 tok/s md5 9449\n'
                            'X1 r1 7 tok/s\nX1 r2 8 tok/s md5 9449ae5738abcdef', 'legs-a.out')
        self.assertIn('md5 gold 1/4', row)
        self.assertIn('missing', row)
        self.assertNotIn('four-pass means', row)
        self.assertNotIn('all 4', row)

    def test_failed_legs_and_aborts(self):
        row = self.fixture('legs-pol.out')
        self.assertIn('POL1 fill_late: expert store: 68251 lookups, hit rate 91.7%', row)
        self.assertIn('POL0 fill_late: expert store: 237519 lookups, hit rate 92.2%', row)
        self.assertNotIn('four-pass means', row)
        self.assertIn('JSONDecodeError:', row)
        rows = self.fixture('runner2-xym.txt').splitlines()
        self.assertEqual(len(rows), 3)
        self.assertIn('ABORT-WINDOW apply failed', rows[0])
        self.assertIn('ABORT-WINDOW 0014 does not apply', rows[1])

    def test_bench_hashes_and_unlabeled_failed_server(self):
        bench = self.fixture('legs-v.out')
        self.assertIn('gemv-build=0', bench)
        self.assertIn('bankread-build=0', bench)
        self.assertIn('other md5s seen', bench)
        self.assertIn('dc8889560109fb82b2858480c6a05117', bench)
        row = self.fixture('runner-p-q-r-s.txt').splitlines()[2]
        self.assertIn('POL0 fill_late: expert store: 237519 lookups', row)
        self.assertIn('unlabeled fill_late: expert store: 68251 lookups', row)
        self.assertNotIn('POL0 fill_late: expert store: 68251', row)

    def test_cli_multiple_files(self):
        result = subprocess.run([sys.executable, str(SCRIPT), str(FIXTURES / 'legs-w.out'),
                                 str(FIXTURES / 'runner3-s-l.txt')], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 4)
        self.assertEqual(result.stderr, '')


if __name__ == '__main__':
    unittest.main(verbosity=2)
