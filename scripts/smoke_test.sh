#!/usr/bin/env bash
# Fast smoke test config for AGX Orin or DGX Spark.
# Purpose: Verify pipeline end-to-end with short run (~5 min).
# Uses tiny batch/seq to fit AGX memory, short wallclock.
#
# Usage:
#   bash scripts/smoke_test.sh /path/to/train_script.py /path/to/data_dir [extra env vars]
#
# Example:
#   DATA_DIR=/mnt/nvme/data bash scripts/smoke_test.sh train_gpt_agx.py
#   TTT_ENABLED=1 bash scripts/smoke_test.sh train_gpt_agx.py

set -e

SCRIPT="${1:?script path required}"
shift

# Smoke test hyperparams (short, light, still representative)
export ITERATIONS="${ITERATIONS:-500}"
export WARMUP_STEPS="${WARMUP_STEPS:-10}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-32768}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export VAL_BATCH_TOKENS="${VAL_BATCH_TOKENS:-131072}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}"
# VAL_LOSS_EVERY=0 → no intermediate/step-0 validation; only final val runs.
# eval_val iterates the FULL 40M-token val set, so intermediate vals are costly on small-batch setups.
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export GPTQ_CALIBRATION_BATCHES="${GPTQ_CALIBRATION_BATCHES:-8}"
export GPTQ_RESERVE_SECONDS="${GPTQ_RESERVE_SECONDS:-5}"
export SLIDING_WINDOW_ENABLED="${SLIDING_WINDOW_ENABLED:-0}"

# Allow all other SOTA hparams to use their defaults unless overridden
# (NUM_LAYERS=11, MODEL_DIM=512, NUM_HEADS=8, etc.)

mkdir -p logs
RUN_ID="${RUN_ID:-smoke_$(date +%Y%m%d_%H%M%S)}"
export RUN_ID

echo "=============================================="
echo "SMOKE TEST: $SCRIPT"
echo "  RUN_ID=$RUN_ID"
echo "  DATA_DIR=$DATA_DIR"
echo "  ITERATIONS=$ITERATIONS  SEQ=$TRAIN_SEQ_LEN  BS=$TRAIN_BATCH_TOKENS"
echo "  MAX_WALLCLOCK=${MAX_WALLCLOCK_SECONDS}s"
echo "  TTT_ENABLED=${TTT_ENABLED:-0}  NUM_LOOPS=${NUM_LOOPS:-2}  LOOP_START=${LOOP_START:-3}  LOOP_END=${LOOP_END:-5}"
echo "=============================================="

exec python3 "$SCRIPT" "$@"
