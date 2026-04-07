#!/usr/bin/env bash
# eval.sh — Immutable evaluation wrapper for Parameter Golf autoresearch
#
# Three modes (mutually exclusive):
#   QUICK=1  — 60s prescreen, no sliding window (~1.5 min total)
#   (default) — DEV mode: 250s train, no sliding window (~5-6 min total)
#   FULL=1   — 600s train + sliding window (~12 min, record submissions only)
#
# NEVER modify this file during experiments.

set -euo pipefail

# Derive seed from git commit hash for reproducibility
COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "0000000")
SEED=${SEED:-$(printf '%d' "0x${COMMIT_HASH:0:8}" 2>/dev/null | head -c 9 || echo 1337)}

if [ "${QUICK:-0}" = "1" ]; then
    echo "=== QUICK SCREEN MODE (60s wallclock) ==="
    MAX_WALLCLOCK_SECONDS=60 \
    VAL_LOSS_EVERY=50 \
    TRAIN_LOG_EVERY=50 \
    EVAL_STRIDE=0 \
    SEED="$SEED" \
    python3 train_gpt.py 2>&1
    echo "=== QUICK SCREEN COMPLETE ==="

elif [ "${FULL:-0}" = "1" ]; then
    echo "=== FULL EVAL MODE (600s wallclock + sliding window) ==="
    MAX_WALLCLOCK_SECONDS=600 \
    SEED="$SEED" \
    python3 train_gpt.py 2>&1
    echo "=== FULL EVAL COMPLETE ==="

else
    echo "=== DEV EVAL MODE (250s wallclock, roundtrip only) ==="
    MAX_WALLCLOCK_SECONDS=250 \
    EVAL_STRIDE=0 \
    SEED="$SEED" \
    python3 train_gpt.py 2>&1
    echo "=== DEV EVAL COMPLETE ==="
fi
