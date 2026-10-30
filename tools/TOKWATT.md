# Token energy reporting

Added stdlib-only `tools/tokwatt.py` and offline regressions in
`tools/test_tokwatt.py`. Reports per decode window: tokens/s, mean cabinet W,
median preceding-two-minute baseline W, incremental W, tokens/incremental J
(equivalently tokens/s/W), whole-cabinet tokens/kWh, sample counts and flags.
Queries Prometheus at 10 s resolution; rejects ambiguous multiple series.
Missing/nonfinite samples are omitted and flagged; unavailable metrics or
nonpositive incremental efficiency denominators produce NA.

Added opt-in `TOKWATT=1` markers to `evidence/glm-legs-v2b.sh` for measured
r1/r2 requests, excluding warmup. Added formulas, marker instructions and
measurement caveats to the energy section in `docs/GLM-STATUS.md`.

Usage (live commands are examples; no network was used):

```sh
python3 tools/tokwatt.py --demo
# PROM_URL is read from the environment; no address is hardcoded.
python3 tools/tokwatt.py --start 1793300000 --end 1793300060 --tokens 300
python3 tools/tokwatt.py --prom-url "$PROM_URL" --legs-log legs.out
# On the Spark, retain full stdout, including LEG markers:
TOKWATT=1 bash evidence/glm-legs-v2b.sh | tee legs.out
```

Markers: `LEG name start=<epoch-seconds> end=<epoch-seconds> tokens=N`.
Explicit start/end inputs describe decode only; fractional epochs are accepted.
Each window has its own preceding baseline. Overlapping decode windows fail.
Partial coverage metrics extrapolate the observed mean across the full duration.

Test commands:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools -p test_tokwatt.py -v
python3 tools/tokwatt.py --demo
bash -n evidence/glm-legs-v2b.sh
git diff --check
```

Last test output lines:

```text
Ran 5 tests in 0.046s

OK
synthetic  60.000  300  5.0000  538.0000  468.0000  70.0000  0.0714  33457.2491  6/6,12/12  ok
missing-sample  60.000  300  5.0000  538.0000  468.0000  70.0000  0.0714  33457.2491  5/6,12/12  missing-decode
DEMO PASS: known energy, missing/NaN samples, absent baseline, zero/negative delta, LEG parsing
```

Uncertainties: Prometheus and Spark were not contacted. Receipt-time markers
estimate decode boundaries from server timings; transport/Python startup shifts
them. Relative historic logs alone cannot recover absolute decode windows.
Back-to-back runs/warmup can contaminate the baseline; arrange two quiet minutes.
The shared cabinet, UPS losses and 10 s cadence limit attribution. Incremental
watts exclude idle draw and understate a dedicated box's total draw. Upstream
Home Assistant outages can remain hidden behind repeated fresh scraped values.
