#!/usr/bin/env bash
# Deeper recurrence: NUM_LOOPS=3 and optional earlier ENABLE_LOOPING_AT.
# Tune wallclock if training exceeds budget.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
NPROC="${NPROC_PER_NODE:-8}"
echo "========== NUM_LOOPS=3 ENABLE_LOOPING_AT=0.30 =========="
NUM_LOOPS=3 ENABLE_LOOPING_AT=0.30 torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN" || true
echo "========== NUM_LOOPS=3 LOOP_START=2 LOOP_END=6 (wider range) =========="
NUM_LOOPS=2 LOOP_START=2 LOOP_END=6 ENABLE_LOOPING_AT=0.30 \
  torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN" || true
