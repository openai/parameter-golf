#!/bin/bash
# Usage: ./run_experiment.sh <experiment_dir> [seed] [wallclock_seconds]
# Example: ./run_experiment.sh idea01_byte_weighted_loss 42 600
#
# For smoke test (2 min): ./run_experiment.sh idea01_byte_weighted_loss 42 120
# For full run (10 min):  ./run_experiment.sh idea01_byte_weighted_loss 42 600
#
# Set CUDA_VISIBLE_DEVICES to pick GPU:  CUDA_VISIBLE_DEVICES=2 ./run_experiment.sh ...
# Set NUM_GPUS=8 for multi-GPU:          NUM_GPUS=8 ./run_experiment.sh ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT="${1:?Usage: $0 <experiment_dir> [seed] [wallclock_seconds]}"
SEED="${2:-42}"
WALLCLOCK="${3:-600}"
EXP_DIR="$SCRIPT_DIR/$EXPERIMENT"

if [ ! -f "$EXP_DIR/train_gpt.py" ]; then
    echo "ERROR: $EXP_DIR/train_gpt.py not found"
    exit 1
fi

source "$REPO_DIR/.venv/bin/activate"

export SEED="$SEED"
export MAX_WALLCLOCK_SECONDS="$WALLCLOCK"
export DATA_PATH="$REPO_DIR/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model"
export RUN_ID="${EXPERIMENT}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

# Smoke test mode: disable expensive sliding window eval
if [ "$WALLCLOCK" -le 300 ]; then
    export EVAL_STRIDE=0          # 0 = standard eval (no sliding window)
    export VAL_LOSS_EVERY=0       # skip periodic val, only final
    export TRAIN_LOG_EVERY=50     # more frequent train logging for short runs
    echo "  [SMOKE MODE] sliding window eval DISABLED, torch.compile ON"
fi

LOG_FILE="$EXP_DIR/train_seed${SEED}.log"

echo "=== Running experiment: $EXPERIMENT ==="
echo "  Seed: $SEED"
echo "  Wallclock: ${WALLCLOCK}s"
echo "  Log: $LOG_FILE"
echo ""

cd "$EXP_DIR"

# Use specified GPU count, default 1
USE_GPUS="${NUM_GPUS:-1}"

echo "  Using $USE_GPUS GPU(s)"

if [ "$USE_GPUS" -eq 1 ]; then
    python train_gpt.py 2>&1 | tee "$LOG_FILE"
else
    torchrun --standalone --nproc_per_node="$USE_GPUS" train_gpt.py 2>&1 | tee "$LOG_FILE"
fi

# Extract final results
echo ""
echo "=== Results for $EXPERIMENT (seed=$SEED) ==="
grep -E "final_int8_zlib_roundtrip" "$LOG_FILE" | tail -1 || echo "  (no final result found)"
grep -E "Serialized model int6" "$LOG_FILE" || echo "  (no artifact size found)"
