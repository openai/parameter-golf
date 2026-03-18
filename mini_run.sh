#!/usr/bin/env bash
# mini_run.sh — fast local architecture testing (no CUDA needed on Mac with DEV_MODE=1)
# Usage:
#   ./mini_run.sh                           # baseline mini-run
#   WINDOW_SIZE=128 ./mini_run.sh           # sliding window attention
#   NUM_PHYSICAL_LAYERS=3 ./mini_run.sh     # weight tying (3 blocks x 3 recurrences)
#   WINDOW_SIZE=128 NUM_PHYSICAL_LAYERS=3 ./mini_run.sh  # both

set -euo pipefail

ITERATIONS="${ITERATIONS:-200}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-50}"
SKIP_QUANT="${SKIP_QUANT:-1}"
WINDOW_SIZE="${WINDOW_SIZE:-0}"
NUM_PHYSICAL_LAYERS="${NUM_PHYSICAL_LAYERS:-0}"
DEV_MODE="${DEV_MODE:-0}"

# Detect if CUDA is available; if not, enable dev mode automatically.
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  || DEV_MODE=1

echo "=== mini_run ==="
echo "  ITERATIONS=$ITERATIONS  VAL_LOSS_EVERY=$VAL_LOSS_EVERY"
echo "  WINDOW_SIZE=$WINDOW_SIZE  NUM_PHYSICAL_LAYERS=$NUM_PHYSICAL_LAYERS"
echo "  SKIP_QUANT=$SKIP_QUANT  DEV_MODE=$DEV_MODE"
echo "================="

ITERATIONS="$ITERATIONS" \
VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
SKIP_QUANT="$SKIP_QUANT" \
WINDOW_SIZE="$WINDOW_SIZE" \
NUM_PHYSICAL_LAYERS="$NUM_PHYSICAL_LAYERS" \
DEV_MODE="$DEV_MODE" \
python train_gpt.py 2>&1 | tee mini_run.log

echo ""
echo "=== Results ==="
grep "val_bpb" mini_run.log | tail -5
