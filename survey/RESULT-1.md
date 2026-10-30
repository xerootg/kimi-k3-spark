Implemented the latency survey kit in survey/; usage and interpretation:
[survey/README.md](survey/README.md).

Run on the laptop:
```bash
./survey/run.sh all
# Or: net | nvme | igpu | e2e
python3 survey/analyze.py survey/results/TIMESTAMP --log /path/to/glm-server.log
```
W defaults to ~/glm53-work; verified MUFU tables and the existing oneAPI
container are required for GPU sections. NUCs use their bundled SYCL runtime.
Raw JSONL, logs, metadata and manifest are pulled to survey/results/TIMESTAMP/.
Each invocation has isolated remote files, process ownership and hard watchdogs.
Spark runs CPU/network probes only; its GLM server and port 8080 are untouched.
Store access is read-only; no host tuning changes are made.

Added driver.py, probe.py, GLM-aware glm_readbench.cpp, prepare.py,
analyze.py and a hardware-free synthetic validation script.
Network covers GLM wire sizes, ping-pong, bulk, Ethernet and discovered TB links.
NVMe covers whole-record and split reads, queue depths, and eight-record batches.
Compute covers fast/exact g53x and g53c benchmarks plus local/remote serveidx.
Analyzer reports p50/p90/p99/max/counts, errors, and conditional budget checks.

Repository gaps: g53x has no disk tier; disk_test.py is absent. prepare.py
builds a separate eight-slot LRU/O_DIRECT survey prototype; cold/LRU results
are labelled accordingly. Production sources are unchanged. g53c is bench-only;
there is no GLM CPU server in this checkout. k3cs uses K3 arithmetic.

Default target is under 20 minutes with a 1,180-second overall watchdog.
Exact arithmetic is too slow for 1,000 samples in every case within that target.
Time-capped cases retain available samples and flag weak p99 estimates.
README includes longer rerun settings to obtain full tail distributions.
The 3 ms budget is explicitly assumed until owner logs/trace establish a window.

Local validation command:
```bash
bash -n survey/run.sh
python3 -m py_compile survey/*.py
python3 survey/prepare.py survey/.validation-build
g++ -std=c++17 -pthread -Itools/bitexact -fsyntax-only survey/glm_readbench.cpp
g++ -std=c++17 -march=alderlake -mavxvnni -pthread -Itools/glm53/ref -fsyntax-only survey/.validation-build/g53c.cpp
python3 survey/validate.py
```
Last output lines (full synthetic analyzer output: survey/validation.txt):
```text
NOTE: Log rows are distributions of reported averages, not individual layer observations.
NOTE: Trace windows are reported separately; choose --budget-us for the required lower-tail window. No automatic substitution of their median.
PASS: synthetic nearest-rank tails, counts, log/trace parsing, fits/exceeds; no hardware probes executed.
```
Unverified: SYCL compilation, liburing compilation, remote lifecycle, hardware
measurements and NUC runtime compatibility. Disk prototype needs real-host review.
No network, hardware benchmarks, commits or pushes were performed here.
