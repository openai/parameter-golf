#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Junkyard Rat copy reduced for fast Shroud architecture-flow tracing.
bash experiments/Shroud/profiles/run_junkyard_rat_mini_shroud.sh
