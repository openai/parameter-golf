#!/usr/bin/env bash
set -euo pipefail

# EXP B: SwiGLU — 2-seed validation (1337, 42)

for SEED in 1337 42; do
    echo ""
    echo "========== EXP B: SwiGLU — seed $SEED =========="
    SEED=$SEED bash exp_b/run.sh
done

echo ""
echo "========== EXP B: 2-seed runs complete =========="
