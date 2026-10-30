#!/usr/bin/env bash
# Alternate baseline/current requests in one process to limit thermal drift.
# G53_BASELINE_REF selects a revision containing the pre-optimization kernel.
set -eu
cd "$(dirname "$0")/../../.."
B=tools/glm53/cpu/build
mkdir -p "$B/tmp"
export TMPDIR="$PWD/$B/tmp"
git show "${G53_BASELINE_REF:-HEAD}:tools/glm53/cpu/g53cpu.cpp" > "$B/baseline.cpp"
python3 - <<'PY'
import os, re, subprocess
from pathlib import Path
cpu = Path('tools/glm53/cpu')
names = sorted(set(re.findall(r'\b(g53cpu_\w+)\s*\(', (cpu/'g53cpu.h').read_text())))
flags = ['-O3', '-g', '-std=c++17', '-pthread', '-march=x86-64-v3',
         *os.getenv('G53_VNNI', '-mavxvnni').split(),
         '-ffp-contract=off', '-fno-fast-math', '-Itools/bitexact', '-I'+str(cpu)]
compiler = os.getenv('CXX', 'g++')
subprocess.run([compiler, *flags, '-Dg53cpu_pool=baseline_pool',
                *['-D'+n+'='+n.replace('g53cpu_', 'baseline_') for n in names],
                '-c', str(cpu/'build/baseline.cpp'), '-o', str(cpu/'build/baseline.o')], check=True)
subprocess.run([compiler, *flags, '-DG53_BENCH_PAIRED', str(cpu/'g53cpu.cpp'),
                str(cpu/'bench.cpp'), str(cpu/'build/baseline.o'),
                '-o', str(cpu/'build/bench-paired')], check=True)
PY
"$B/bench-paired" "${1:-8}" "${2:-31}"
