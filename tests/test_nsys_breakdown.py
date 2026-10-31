"""Synthetic, offline coverage for nsys 2025.3-style exports."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/glm53/nsys_breakdown.py'
SPEC = importlib.util.spec_from_file_location('nsys_breakdown', SCRIPT)
nsys = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nsys)


class BreakdownTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.directory = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, text):
        path = self.directory / name
        path.write_text(text, encoding='utf-8')
        return path

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), str(self.directory), *args],
                              text=True, capture_output=True)

    def test_banner_bom_reordered_columns_and_quoted_number(self):
        path = self.write('cuda_gpu_kern_sum.csv', '\ufeff# nsys 2025.3\n\nGenerating report...\n'
                          'Name,Instances,Total Time (ns),Time (%)\n'
                          '"void mul_mat_vec_q<int, float>()",2,"2,000,000",100\n')
        rows = nsys.read_csv(str(path))
        self.assertEqual(nsys.totals(rows, nsys.kernel_family)[nsys.DENSE], 2000000)

    def test_families_and_shared_precedence(self):
        names = ['mul_mat_vec_q<(ggml_type)17>', 'mul_mat_vec_q<Q2>',
                 'shared_mmvq', 'flash_attn_ext', 'indexer_topk',
                 'rms_norm', 'rope_f32', 'get_rows_q', 'some_gemm']
        self.assertEqual([nsys.kernel_family(n) for n in names], list(nsys.FAMILIES))
        self.assertEqual(nsys.kernel_family('mmvq_kernel'), nsys.DENSE)
        for kind in ('17', '18', '23', 'IQ2_XS', 'IQ3_XXS', 'IQ4_XS'):
            name = 'mul_mat_vec_q<' + kind + '>'
            self.assertEqual(nsys.kernel_family(name, 9), nsys.BANK)
            self.assertEqual(nsys.kernel_family(name, 8), nsys.BANK)
            self.assertEqual(nsys.kernel_family(name, 1), nsys.DENSE)
        self.assertEqual(nsys.grid({'Name': 'mul_mat_vec_q<17> grid2048x9'}, 'Y'), 9)
        self.assertEqual(nsys.grid({'Name': 'mul_mat_vec_q<18> grid6144x8'}, 'X'), 6144)
        self.assertEqual(nsys.kernel_family('mul_mat_vec_q<(ggml_type)18>', 8), nsys.BANK)
        for name in ('dsa_kernel', 'topk_kernel'):
            self.assertEqual(nsys.kernel_family(name), 'DSA indexer')

    def test_gap_boundaries_overlap_layer_and_device(self):
        rows = []
        start = 0
        # Boundaries: 50us belongs to second, 500us to third, 5ms to third.
        gaps = [49999, 50000, 499999, 500000, 5000000, 5000001]
        for gap in [0] + gaps:
            start += gap
            rows.append({'Start (ns)': str(start), 'Duration (ns)': '1000',
                         'Name': 'kernel', 'Device ID': '0', 'Layer ID': '7'})
            start += 1000
        # A contained concurrent kernel must not shift the frontier backwards.
        rows.append({'Start (ns)': '100', 'Duration (ns)': '100', 'Name': 'other',
                     'Device ID': '0', 'Layer ID': '7'})
        rows.append({'Start (ns)': '0', 'Duration (ns)': '10', 'Name': 'kernel',
                     'Device ID': '1', 'Layer ID': '7'})
        busy, span, buckets, long_gaps = nsys.trace_metrics(list(reversed(rows)))
        self.assertEqual(busy, 7010)
        self.assertEqual(span - busy, sum(gaps))
        self.assertEqual(long_gaps, 10000001)
        self.assertEqual(buckets[('0', '7')],
                         [[1, 49999], [2, 549999], [2, 5500000], [1, 5000001]])

    def test_trace_end_variant_and_copy_filter(self):
        path = self.write('cuda_gpu_trace.csv', '# comment\n\n'
                          'Kernel Name,End Time (us),Start Time (us),Device\n'
                          'kernel_a,10,0,0\n'
                          '[CUDA memcpy HtoD],50,10,0\n'
                          'kernel_b,70,60,0\n')
        busy, span, buckets, long_gaps = nsys.trace_metrics(nsys.read_csv(str(path)))
        self.assertEqual((busy, span), (20000, 70000))
        self.assertEqual(buckets[('0', 'unassigned')][1], [1, 50000])

    def test_complete_win2m_reports_summary_and_comparison(self):
        for prefix, scale in [('glm-short', 1), ('glm-long', 2)]:
            self.write(prefix + '_cuda_gpu_kern_sum.csv',
                       'Time (%),Total Time (us),Instances,Name\n'
                       '100,{},1,mmvq_kernel\n'.format(2000 * scale))
            self.write(prefix + '_cuda_gpu_trace.csv',
                       'Start (ms),Duration (ms),Name\n'
                       '0,{},mmvq_kernel\n{}, {},mmvq_kernel\n'.format(
                           scale, 43 * scale, scale))
            self.write(prefix + '_cuda_api_sum.csv',
                       'Name,Total Time (ms),Num Calls\n'
                       'cuStreamWaitValue32,{},1\n'
                       'cudaGraphLaunch_v10000,1,1\n'
                       'cudaMemcpyAsync,1,1\n'
                       'cudaEventSynchronize,1,1\n'
                       'cudaFree,1,1\n'.format(42 * scale))
        ambiguous = self.run_cli()
        self.assertNotEqual(ambiguous.returncode, 0)
        self.assertIn('multiple traces', ambiguous.stderr)
        result = self.run_cli(str(self.directory), '--prefix', 'glm-short',
                              '--prefix2', 'glm-long', '--tokens', '1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('tokens=1 gpu_busy=2.000000 ms/token gaps=42.000000 ms/token '
                      '(of which >0.5ms: 42.000000) api=46.000000 ms/token', result.stdout)
        self.assertIn('gpu_busy=4.000000 ms/token gaps=84.000000', result.stdout)
        self.assertIn('Delta (second - first)', result.stdout)
        self.assertIn('delta +0.000', result.stdout)
        self.assertIn('4.55%', result.stdout)

    def test_missing_reports_default_tokens_and_custom_prefix(self):
        self.write('custom_cuda_api_sum.csv',
                   'Name,Total Time (ns),Instances\ncudaGraphLaunch,384000000,1\n')
        result = self.run_cli('--prefix', 'custom')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('default: 128 x 3 requests', result.stdout)
        self.assertIn('Missing reports: cuda_gpu_kern_sum, cuda_gpu_trace', result.stdout)
        self.assertIn('gpu_busy=n/a ms/token gaps=n/a', result.stdout)
        self.assertIn('api=1.000000 ms/token', result.stdout)

    def test_summary_only_and_empty_directory(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = nsys.analyze(str(self.directory), 64)
        self.assertIsNone(result['busy'])
        self.assertIn('Missing reports: cuda_gpu_kern_sum, cuda_gpu_trace, cuda_api_sum', output.getvalue())
        self.write('cuda_gpu_kern_sum.csv', 'Name,Total Time (ns)\nrope,64000000\n')
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = nsys.analyze(str(self.directory), 64)
        self.assertEqual(result['busy'], 1)
        self.assertIn('n/a | 100.00%', output.getvalue())

    def test_grid_classification_split_and_gap_attribution(self):
        self.write('cuda_gpu_kern_sum.csv',
                   'Name,Total Time (ms),Instances\n'
                   'mul_mat_vec_q<17>,4,3\n'
                   'mul_mat_vec_q<IQ3_XXS>,2,1\n'
                   'mul_mat_vec_q<13>,1,1\n')
        self.write('cuda_gpu_trace.csv',
                   'Name,Start (ms),Duration (ms),GrdX,GrdY,Device\n'
                   'mul_mat_vec_q<13>,0,1,2048,1,0\n'
                   'request,1,1,1,1,0\n'
                   'mul_mat_vec_q<17>,3,1,2048,9,0\n'
                   'mul_mat_vec_q<IQ3_XXS>,4,2,6144,9,0\n'
                   'request,6,1,1,1,0\n'
                   'gather,9,1,1,1,0\n'
                   'mul_mat_vec_q<17>,10,1,2048,8,0\n'
                   'mul_mat_vec_q<17>,11,2,2048,1,0\n')
        result = self.run_cli('--tokens', '2', '--decode-passes', '1')
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
                'host-bank expert GEMV: 2.000000',
                'dense-spine GEMV (device): 1.500000',
                'early (Y=9): 1.500000 | 3.000000 | 2 | 1.000000 | 2.000000',
                'late (Y=8): 0.500000 | 1.000000 | 1 | 0.500000 | 1.000000',
                'pre-first-bank-kernel: 1 | 0.500000 | 1.000000',
                'early-to-late: real-lookup wait (6144-wide early down): 1 | 1.000000 | 2.000000'):
            self.assertIn(expected, result.stdout)
        self.assertNotIn('Caveat:', result.stdout)

    def test_summary_grid_and_type_only_caveat(self):
        self.write('cuda_gpu_kern_sum.csv',
                   'Name,Total Time (ms),Instances,Grid Y\n'
                   'mul_mat_vec_q<17>,1,1,1\n'
                   'mul_mat_vec_q<18>,2,2,9\n'
                   'mul_mat_vec_q<IQ4_XS>,3,3,\n')
        result = self.run_cli('--tokens', '1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Caveat:', result.stdout)
        self.assertIn('host-bank expert GEMV: 5.000000', result.stdout)
        self.assertIn('dense-spine GEMV (device): 1.000000', result.stdout)
        self.assertIn('unknown Y: 3.000000', result.stdout)
        self.assertNotEqual(self.run_cli('--decode-passes', '0').returncode, 0)

    def test_invalid_tokens_and_malformed_reports(self):
        result = self.run_cli('--tokens', '0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must be positive', result.stderr)
        self.write('cuda_api_sum.csv', 'this is not a report\n')
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no recognizable CSV header', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
