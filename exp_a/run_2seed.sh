#!/usr/bin/env bash
set -euo pipefail

# EXP A: MTP — 2-seed validation (1337, 42)

for SEED in 1337 42; do
    echo ""
    echo "========== EXP A: MTP — seed $SEED =========="
    SEED=$SEED bash exp_a/run.sh
done

echo ""
echo "========== EXP A: 2-seed runs complete =========="
