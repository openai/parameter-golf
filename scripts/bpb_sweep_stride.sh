#!/usr/bin/env bash
# Phase 1d: EVAL_STRIDE sweep (e.g. 64 vs 32). Check eval stays within 600s budget.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
NPROC="${NPROC_PER_NODE:-8}"
for STRIDE in 64 32; do
  echo "========== EVAL_STRIDE=$STRIDE =========="
  EVAL_STRIDE="$STRIDE" torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN" || true
done
