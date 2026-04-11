#!/usr/bin/env bash
set -euo pipefail

# EXP C: Vocab 1536 — 2-seed validation (1337, 42)

for SEED in 1337 42; do
    echo ""
    echo "========== EXP C: Vocab 1536 — seed $SEED =========="
    SEED=$SEED bash exp_c/run.sh
done

echo ""
echo "========== EXP C: 2-seed runs complete =========="
