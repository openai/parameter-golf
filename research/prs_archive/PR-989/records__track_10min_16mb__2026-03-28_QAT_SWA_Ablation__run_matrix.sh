#!/usr/bin/env bash
set -euo pipefail

# Run the full 2x2 matrix with 2 seeds each (8 total runs)
# Usage: bash run_matrix.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"

EXPERIMENTS=(control qat_snap70 no_swa no_swa_qat)
SEEDS=(42 1337)

for exp in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "============================================"
        echo "  Running: $exp (seed=$seed)"
        echo "============================================"
        bash "$SCRIPT_DIR/run.sh" "$exp" "$seed"
    done
done

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================"
echo "Logs in: $SCRIPT_DIR/logs/"
