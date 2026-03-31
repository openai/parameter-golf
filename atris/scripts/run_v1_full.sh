#!/bin/bash
# Atris v1 FULL: Modified train_gpt.py with all code-level improvements
# Uses v1_train_gpt.py which has:
#   1. 10 layers (was 9)
#   2. Lower LRs (0.02/0.02/0.03)
#   3. INT6 mixed precision for middle layers (saves ~1.6MB)
#   4. Eval at configurable seq length (set EVAL_SEQ_LEN=2048 for longer context)
#
# Run on 8xH100 for final submission, or 1 GPU for dev

set -euo pipefail

cd "$(dirname "$0")/../.."

# Copy our modified script to train_gpt.py (backup original first)
if [ ! -f train_gpt.py.orig ]; then
    cp train_gpt.py train_gpt.py.orig
fi
cp atris/experiments/v1_train_gpt.py train_gpt.py

NPROC=${NPROC:-8}
WALLCLOCK=${WALLCLOCK:-600}
EVAL_SEQ=${EVAL_SEQ_LEN:-1024}

echo "================================================"
echo "  ATRIS v1 FULL: All code improvements"
echo "  GPUs: $NPROC | Wallclock: ${WALLCLOCK}s"
echo "  Eval seq len: $EVAL_SEQ"
echo "================================================"

NCCL_IB_DISABLE=1 \
RUN_ID="atris_v1_full_$(date +%s)" \
EVAL_SEQ_LEN=$EVAL_SEQ \
MAX_WALLCLOCK_SECONDS=$WALLCLOCK \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee atris/logs/v1_full_run.log

echo ""
echo "================================================"
echo "  Run complete. Key metrics:"
grep -E "(final_int8_zlib_roundtrip|submission size)" atris/logs/v1_full_run.log || true
echo "================================================"

# Also try with longer eval context
if [ "$EVAL_SEQ" = "1024" ]; then
    echo ""
    echo "TIP: Try EVAL_SEQ_LEN=2048 for potentially free BPB improvement:"
    echo "  EVAL_SEQ_LEN=2048 bash atris/scripts/run_v1_full.sh"
fi
