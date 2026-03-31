#!/bin/bash
# Atris v1 DEV: Quick iteration on 1 GPU (2 min runs)
# Use this for fast testing before burning 8xH100 time
#
# Cost: ~$0.05 per run on 1xA100

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "================================================"
echo "  ATRIS v1 DEV: Quick test (1 GPU, 2 min)"
echo "================================================"

NCCL_IB_DISABLE=1 \
RUN_ID="atris_v1_dev_$(date +%s)" \
NUM_LAYERS=10 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MAX_WALLCLOCK_SECONDS=120 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee atris/logs/v1_dev_run.log

echo ""
echo "Check final val_bpb above."
echo "If it looks promising, run run_v1.sh on 8xH100."
