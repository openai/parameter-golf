#!/bin/bash
# Atris v1: Proven hyperparameter improvements (no code changes needed)
# All changes via environment variables
#
# Changes from baseline:
#   NUM_LAYERS: 9 → 10 (nanlliu, PR #39)
#   MATRIX_LR: 0.04 → 0.02 (consensus from multiple submissions)
#   SCALAR_LR: 0.04 → 0.02
#   TIED_EMBED_LR: 0.05 → 0.03
#
# Expected: ~1.21-1.22 BPB (beats baseline 1.2244)
# Cost: ~$3.60 per run on 8xH100

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "================================================"
echo "  ATRIS v1: Tuned Hyperparameters"
echo "  Target: < 1.22 BPB"
echo "================================================"

NCCL_IB_DISABLE=1 \
RUN_ID="atris_v1_$(date +%s)" \
NUM_LAYERS=10 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=${NPROC:-8} train_gpt.py 2>&1 | tee atris/logs/v1_run.log

echo ""
echo "================================================"
echo "  Run complete. Check val_bpb above."
echo "================================================"
