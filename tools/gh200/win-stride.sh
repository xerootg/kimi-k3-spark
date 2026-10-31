#!/usr/bin/env bash
# Run locally on gh200-dev with kubectl access, only after outage authorization.
# LOUD: docs/gh200-dev.md and configs/ are absent. Historical ~/llama-k3-la
# (experiments.md:590) does NOT identify a verified current native checkout.
# Set NATIVE_BASE to a verified offline checkout containing versions.env's SHA.
# LOUD: the ac78af0e5d request JSON is NOT present in docs/tools. Supply the
# owner's exact K3_REQUEST_JSON; do not substitute the Spark quicksort fixture.
set -euo pipefail
exec python3 "$(dirname -- "$0")/win-stride.py" "$@"
