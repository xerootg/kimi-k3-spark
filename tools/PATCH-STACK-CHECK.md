Changed:
- Added .github/workflows/patch-stack.yml for pushes and PRs touching patches/,
  versions.env, the checker, or the workflow itself.
- CI installs liburing-dev, shallow-fetches only LLAMA_SHA from ggml-org/llama.cpp,
  then runs the shared checker. The job has a two-minute timeout.
- Added executable tools/patch-stack-check.sh for a local llama.cpp checkout,
  e.g. tools/patch-stack-check.sh ~/llama-k3-s2 on the Spark.
- Local HEAD must match versions.env. A temporary copy of committed HEAD is
  checked; dirty files and the supplied checkout are left untouched. No fetch.
- Both environments check then apply every patch in lexical order, reporting
  the patch name on rejection. Two concurrent c++ -fsyntax-only commands check
  touched .cpp files under src/ (including models/) and ggml/src/ggml-cpu/.
  Include paths cover llama public/internal/model and ggml public/internal/CPU
  headers. C, CUDA, and standalone headers are excluded; no full build runs.
- Updated patches/README.md with cumulative patch order and check instructions.

Validation (offline, all passed):
- bash -n tools/patch-stack-check.sh
  Last output: none (exit 0).
- python -c 'import yaml; p=".github/workflows/patch-stack.yml";
  d=yaml.safe_load(open(p)); assert d["jobs"]["check"]["timeout-minutes"] == 2;
  print(p + ": YAML valid")'
  Last output: .github/workflows/patch-stack.yml: YAML valid
- git diff --check
  Last output: none (exit 0).
- python - <<'PY' ... PY (temporary offline Git fixture harness).
  Copied the checker into a fixture with two cumulative patches, nested model
  and CPU C++ files, and intentionally invalid C/CUDA files. Ran success,
  rejection, C++ syntax failure, wrong-SHA, dirty-tree preservation, and cleanup
  assertions using the real Git and C++ compiler.
  Last output lines (ERROR lines below were expected and asserted):
  PASS: 2 patches applied; 3 C++ files checked
  PASS: input checkout local changes preserved; C/CUDA skipped
  ERROR: patch rejected: 0002.patch
  ERROR: syntax check failed: src/test.cpp
  PASS: wrong SHA rejected; temporary trees cleaned after success and failures

Unverified:
- No upstream llama.cpp tree exists inside this directory. The actual stack
  and its include requirements were not compiled here; network was not used.
- CI fetch/install/compile wall time is unmeasured; two minutes is enforced
  by the job timeout, not a measured successful duration.
- This checkout contains 0001 through 0006, not the mentioned 0007/0010.
  The checker discovers future patches and touched C++ files automatically.
