"""Synthetic TSV regression tests for prefill/decode alignment and failures."""
from pathlib import Path
import tempfile
import unittest
from prefill_compare import summary


def row(step, name, op='ADD', col=0, digest='0'*16, dims='4,1,1,1', axis=3, value=''):
    return f'{step}\t0\t{name}\t{op}\tf32\t{dims}\t{axis}\t{col}\t{digest}\t{value}\n'


def fixture(col=0, offset=0):
    text = ''
    for step in range(3):
        s = step + offset
        if step:
            text += row(s, 'norm-0', 'RMS_NORM', col)
            if col:
                text += row(s, 'cache_k_l0', 'RESHAPE', col, digest='f'*16)
                text += row(s, 'node_99', 'CPY', col)
            text += row(s, 'norm-0', 'MUL', col)  # repeated name, distinct op
            text += row(s, 'moe_nowait_merge-3', 'GET_ROWS', col)
            for size in (9, 8):
                text += row(s, 'ffn_moe_gate-3', 'MUL_MAT_ID', col,
                            digest=('f' if col else '0')*16, dims=f'4,{size},1,1')
            text += row(s, 'l_out-3', col=col)
        text += row(s, '@logits', col=col, digest=str(step)*16)
        text += row(s, '@sampled_token', col=col, value=str(100+step))
    return text


class PrefillCompare(unittest.TestCase):
    def run_pair(self, a, b):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            gold, batch = Path(tmp)/'a.tsv', Path(tmp)/'b.tsv'
            gold.write_text(a); batch.write_text(b)
            out = []
            rc = summary(gold, batch, 1, out.append)
            return rc, '\n'.join(out)

    def test_resync_steps_columns_and_expected_scratch(self):
        rc, out = self.run_pair(fixture(), fixture(1, 5))
        self.assertEqual(rc, 0)
        self.assertIn('PREFILL column=1: 2/2', out)
        self.assertIn('steps=1/6: logits', out)
        self.assertIn('layer=0: 2/2 compute matches (100.00%)', out)
        self.assertIn('2 expected split-bank', out)

    def test_real_compute_diff_fails(self):
        b = fixture(1).replace('l_out-3\tADD\tf32\t4,1,1,1\t3\t1\t'+'0'*16,
                                'l_out-3\tADD\tf32\t4,1,1,1\t3\t1\t'+'a'*16)
        self.assertEqual(self.run_pair(fixture(), b)[0], 1)

    def test_missing_compute_resynchronises_and_fails(self):
        b = ''.join(line for line in fixture(1).splitlines(True) if 'norm-0\tMUL' not in line)
        rc, out = self.run_pair(fixture(), b)
        self.assertEqual(rc, 1)
        self.assertIn('ALIGNMENT step=1', out)
        self.assertIn('layer=3: 1/3', out)

    def test_logits_prefill_and_sample_count_fail(self):
        for b in (fixture(1).replace('1'*16, 'a'*16),
                  fixture(1).replace('0'*16, 'a'*16),
                  ''.join(l for l in fixture(1).splitlines(True) if not l.startswith('2\t'))):
            self.assertEqual(self.run_pair(fixture(), b)[0], 1)

    def test_scratch_requires_split_evidence(self):
        b = fixture(1).replace('moe_nowait_merge-3', 'other-3')
        self.assertEqual(self.run_pair(fixture(), b)[0], 1)


if __name__ == '__main__':
    unittest.main()
