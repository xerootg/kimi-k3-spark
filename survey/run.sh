#!/usr/bin/env bash
# Laptop entry point; all remote work has its own timeout and owned-child cleanup.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# == 0 ]]; then set -- all; fi
exec timeout --signal=TERM --kill-after=15s "${SURVEY_TOTAL_SECONDS:-1180}s" python3 survey/driver.py "$@"
