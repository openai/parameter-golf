#!/usr/bin/env bash
set -euo pipefail

# EXP D: TTT 8ep + stride 32 — 2-seed validation (1337, 42)

for SEED in 1337 42; do
    echo ""
    echo "========== EXP D: TTT8 + stride32 — seed $SEED =========="
    SEED=$SEED bash exp_d/run.sh
done

echo ""
echo "========== EXP D: 2-seed runs complete =========="
