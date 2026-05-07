#!/usr/bin/env bash
# 3-seed confirmation (42, 314, 999). Export other env (TTT, QK, etc.) before running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
NPROC="${NPROC_PER_NODE:-8}"
for SEED in 42 314 999; do
  echo "========== SEED=$SEED =========="
  SEED="$SEED" torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN"
done
