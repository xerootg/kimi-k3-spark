import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('tensor_firstdiff', Path(__file__).with_name('tensor_firstdiff.py'))
fd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fd
spec.loader.exec_module(fd)


def record(name, col=0, digest='0'*16, shape='8,1,1,1', axis=1, step=1):
    return f'{step}\t0\t{name}\tADD\tf32\t{shape}\t{axis}\t{col}\t{digest}\t\n'


class FirstDiff(unittest.TestCase):
    def setUp(self):
        # Keep fixtures inside the sandbox; TemporaryDirectory cleans via Python, no shell rm.
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.addCleanup(self.tmp.cleanup)
        self.a = Path(self.tmp.name)/'n1.tsv'
        self.b = Path(self.tmp.name)/'n2.tsv'

    def run_compare(self, a, b, col=1):
        self.a.write_text(a)
        self.b.write_text(b)
        out = []
        code = fd.compare(self.a, self.b, col, out.append)
        return code, '\n'.join(out)

    def test_first_diff_and_three_preceding(self):
        names = ['input', 'norm', 'q', 'rope', 'attention', 'output']
        a = ''.join(record(n) for n in names)
        b = ''.join(record(n, c, 'f'*16 if n == 'attention' and c == 1 else '0'*16,
                           '8,2,1,1') for n in names for c in (0, 1))
        code, out = self.run_compare(a, b)
        self.assertEqual(code, 1)
        self.assertIn('FIRST-DIFF', out)
        self.assertIn('attention', out)
        self.assertEqual(out.count('MATCH N1'), 3)
        self.assertNotIn('input', out)
        self.assertNotIn('output', out)
        self.assertEqual(self.run_compare(a, b, 0)[0], 0)

    def test_shared_strided_shape_and_repeated_names(self):
        a = record('x', axis=3) + record('x', axis=3)
        b = record('x', 1, shape='8,2,1,1')*2
        self.assertEqual(self.run_compare(a, b)[0], 0)

    def test_missing_node_is_alignment_failure(self):
        code, out = self.run_compare(record('x')+record('y'), record('y', 1))
        self.assertEqual(code, 1)
        self.assertIn('ALIGNMENT', out)

    def test_shape_failure(self):
        self.assertIn('SHAPE/TYPE', self.run_compare(record('x'), record('x', 1, shape='9,2,1,1'))[1])

    def test_prefill_logits_and_step_alignment(self):
        a = record('@logits', step=0)+record('@sampled_token', step=0)+record('norm')
        b = record('@logits', 1, digest='a'*16, step=0)+record('@sampled_token', 1, step=0)+record('norm', 1)
        code, out = self.run_compare(a, b)
        self.assertEqual(code, 1)
        self.assertIn('step=0 @logits', out)
        self.assertNotIn('step=1', out)

    def test_shared_and_scalar_column(self):
        a = record('shared', -1, axis=-1)+record('scalar', shape='1,1,1,1', axis=3)
        b = record('shared', -1, axis=-1)+record('scalar', 1, shape='2,1,1,1', axis=0)
        self.assertEqual(self.run_compare(a, b)[0], 0)

    def test_skipped_large_is_unobserved(self):
        a = record('large').rstrip('\n') + 'skipped-large bytes=70000000\n' + record('norm')
        b = record('large', 1, shape='8,2,1,1') + record('norm', 1)
        code, out = self.run_compare(a, b)
        self.assertEqual(code, 0)
        self.assertIn('UNOBSERVED', out)
        self.assertIn('1 skipped-large pairs unobserved', out)
        b = record('large', 1, shape='8,2,1,1') + record('norm', 1, digest='f'*16)
        self.assertIn('FIRST-DIFF', self.run_compare(a, b)[1])

    def test_bad_and_missing_column(self):
        self.a.write_text(record('x'))
        self.b.write_text(record('x'))
        with self.assertRaises(ValueError): fd.compare(self.a, self.b, 1)
        self.b.write_text('truncated\n')
        with self.assertRaises(ValueError): fd.compare(self.a, self.b, 0)


class EvidenceAlignment(unittest.TestCase):
    def test_real_staggered_prefill_both_modes(self):
        root = Path(__file__).resolve().parents[2]
        for gold, batch in [('FD10', 'FD20'), ('FD11', 'FD21')]:
            for col in (0, 1):
                out = []
                code = fd.compare(root/f'evidence/fd-digests/{gold}.tsv',
                                  root/f'evidence/fd-digests/{batch}.tsv', col, out.append)
                text = '\n'.join(out)
                self.assertEqual(code, 1)
                self.assertIn('FIRST-DIFF COMPUTE', text)
                self.assertIn('indexer_k-0', text)
                self.assertIn('BOOKKEEPING', text)
                self.assertEqual(text.count('MATCH N1'), 3)
                if col:
                    self.assertIn('N1 step=1 token=468 N2 step=2 token=468', text)

if __name__ == '__main__':
    unittest.main()
