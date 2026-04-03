#!/usr/bin/env bash
# gate.sh — runs the full 4-arm legal suite (this IS the gate)
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_suite.sh" "$@"
