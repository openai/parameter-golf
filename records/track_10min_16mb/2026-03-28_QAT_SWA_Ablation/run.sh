#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run.sh <experiment_name> [seed]
# Experiments: control, qat_snap70, no_swa, no_swa_qat
# Default seed: 42

EXPERIMENT="${1:?Usage: bash run.sh <experiment_name> [seed]}"
SEED="${2:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Common settings (identical across all experiments)
export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export SEED="$SEED"
export RUN_ID="${EXPERIMENT}_seed${SEED}"
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100

# Defaults from #180 (unchanged)
export NUM_LAYERS=10
export MODEL_DIM=512
export MLP_MULT=3.0
export NUM_HEADS=8
export NUM_KV_HEADS=4
export WEIGHT_DECAY=0.04
export MUON_MOMENTUM=0.99
export GRAD_CLIP_NORM=0.3
export EVAL_STRIDE=64
export BIGRAM_VOCAB_SIZE=10240
export BIGRAM_DIM=128
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432

# 2x2 Matrix: QAT (on/off) x SWA (on/off)
case "$EXPERIMENT" in
    control)
        # Cell [QAT=off, SWA=on] — this is the #180 baseline
        export ENABLE_QAT=0
        export SWA_ENABLED=1
        ;;
    qat_snap70)
        # Cell [QAT=on, SWA=on] — QAT activates in last 70% of warmdown
        export ENABLE_QAT=1
        export QAT_START_FRAC=0.7
        export SWA_ENABLED=1
        ;;
    no_swa)
        # Cell [QAT=off, SWA=off]
        export ENABLE_QAT=0
        export SWA_ENABLED=0
        ;;
    no_swa_qat)
        # Cell [QAT=on, SWA=off] — QAT without SWA
        export ENABLE_QAT=1
        export QAT_START_FRAC=0.7
        export SWA_ENABLED=0
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Valid: control, qat_snap70, no_swa, no_swa_qat"
        exit 1
        ;;
esac

echo "=========================================="
echo "Experiment: $EXPERIMENT (seed=$SEED)"
echo "ENABLE_QAT=${ENABLE_QAT} SWA_ENABLED=${SWA_ENABLED}"
echo "=========================================="

# Detect GPU count
NGPU=$(nvidia-smi -L | wc -l)
echo "GPUs detected: $NGPU"

mkdir -p "$SCRIPT_DIR/logs"

cd "$REPO_ROOT"
torchrun --standalone --nproc_per_node="$NGPU" \
    experiments/alex_qat/train_gpt.py \
    2>&1 | tee "experiments/alex_qat/logs/${EXPERIMENT}_seed${SEED}.log"
