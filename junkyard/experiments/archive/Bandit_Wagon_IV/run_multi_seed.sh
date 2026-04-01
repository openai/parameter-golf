#!/bin/bash
set -euo pipefail
# ================================================================
#  Bandit_Wagon_IV — MULTI-SEED PRODUCTION
#
#  Runs seeds 444 and 300 sequentially.
#  Run seed 444 first (primary), 300 second (confirmation).
#
#  Usage:
#    NPROC_PER_NODE=8 bash experiments/Bandit_Wagon_IV/run_multi_seed.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for SEED in 444 300; do
    echo ""
    echo "==============================="
    echo "  Starting seed=${SEED}"
    echo "==============================="
    SEED="${SEED}" bash "${SCRIPT_DIR}/run.sh"
done

echo ""
echo "Multi-seed run complete. Check results/ for logs."
