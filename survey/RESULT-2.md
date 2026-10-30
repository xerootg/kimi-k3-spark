Second pass complete; raw owner results and survey-run.log preserved.

Diagnosis: all 76 reported errors reconciled in survey/DIAGNOSIS.md:
- 52 NVMe rc=127 launches: both NUCs lack g++; no binary was produced.
- 16 g53x bench aborts: SYCL could not access a GPU outside render-group sudo.
- 8 g53c failures: laptop binary requires GLIBC_2.43, absent on the NUCs.
Two additional build failures live in .log files, outside the original 76.
Metadata nonzero commands and the layer-78 warning are also explained there.

Changed:
- Build/stage static CPU and NVMe executables; remote g++ is optional.
  Optional successful native liburing build replaces static pread fallback.
  Failed executable preflight skips the sweep instead of cascading 26 errors.
- GPU benches use the successful serveidx render-group/runtime environment.
  Failed/timeout bench logs retain diagnostics.
- Verify all three pre-staged remote MUFU files; removed laptop table requirement.
- Bench cap 65 -> 20 seconds so fixed benches fit the section watchdog.
- Analyzer checks each host/mode/tier/k, reports missing NVMe and error details,
  and reserves budget verdicts for compute e2e rather than network echoes.
- Added offline regressions and updated survey/README.md.
- Wrote survey/FINDINGS.md, DIAGNOSIS.md and full MEASUREMENTS.md.

All 12 Spark sessions and all 96 e2e distributions actually exist; slow cases
are time-capped, not missing. Resident/LRU counters have zero reads; every cold
session has reads. Native benches and isolated NVMe measurements are missing.
Fast small-k resident offload looks plausible; k=8, exact mode 1 and synchronous
cold misses exceed the ~3.1 ms planning budget. Mode 2 remains unmeasured here.
4.9 tok/s / 75 layers implies 2.72 ms mean spacing, not a measured deadline.

Test command (from checkout root):
```bash
set -e
bash -n survey/run.sh
python3 -m py_compile survey/*.py
python3 survey/prepare.py survey/.validation-build
g++ -O2 -static -std=c++17 -pthread -Itools/bitexact survey/glm_readbench.cpp -o survey/.validation-build/glm_readbench
g++ -O2 -static -std=c++17 -march=alderlake -mavxvnni -pthread -Itools/glm53/ref survey/.validation-build/g53c.cpp -o survey/.validation-build/g53c
python3 survey/validate.py > survey/validation.txt
python3 survey/validate_second_pass.py
file survey/.validation-build/glm_readbench survey/.validation-build/g53c
```
Last output lines (ELF details abbreviated):
```text
PASS: all 76 errors reconciled; all 12 disk counter gates verified.
PASS: static NVMe staging, render bench invocation, analyzer coverage and network budget regression; no network used.
survey/.validation-build/glm_readbench: ELF 64-bit ... statically linked ...
survey/.validation-build/g53c:          ELF 64-bit ... statically linked ...
```

Unverified: SYCL build, new hardware runs, render-group bench success, actual
NVMe behavior and NUC runtime compatibility. Rebuild; do not reuse old CPU
artifacts with --skip-build. Exact mode 2 and live GLM deadline/overlap need
owner measurements. No network, commits, pushes or writes outside this directory.
