#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m krel.pipeline --run-id mdace_cm_discharge_gpt4o --label-space full "$@"
