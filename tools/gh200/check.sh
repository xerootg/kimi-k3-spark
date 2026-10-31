#!/usr/bin/env bash
# Offline only. The supplied source is copied here; never modify that reference.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
python3 tools/gh200/prepare-check.py
python3 tools/gh200/check.py
bash -n tools/gh200/win-stride.sh
python3 - <<'PY'
import ast
from pathlib import Path
for p in Path('tools/gh200').glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
for path, limit in [('RESULT-STRIDE.md',40), ('RESULT.md',60)]:
    assert len(Path(path).read_text().splitlines()) <= limit
print('PASS: shell/Python syntax and result line limits; no hardware or network tests')
PY
