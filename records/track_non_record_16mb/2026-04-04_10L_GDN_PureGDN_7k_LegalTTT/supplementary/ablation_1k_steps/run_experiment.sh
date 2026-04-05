#!/usr/bin/env bash
# run_experiment.sh — Run training + evaluation for an experiment.
#
# Usage: ./run_experiment.sh <experiment_name> [arch_mode]
# Example: ./run_experiment.sh exp00_baseline A
#
# The arch_mode argument (default: A) selects the model config from configs.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [arch_mode]"
    echo "Example: $0 exp00_baseline A"
    exit 1
fi

EXPERIMENT_NAME="$1"
ARCH_MODE="${2:-A}"
EXPERIMENT_DIR="${SCRIPT_DIR}/${EXPERIMENT_NAME}"

if [ ! -d "$EXPERIMENT_DIR/code" ]; then
    echo "ERROR: Experiment code dir not found: $EXPERIMENT_DIR/code"
    echo "       Run setup_experiment.sh first."
    exit 1
fi

# ─── Source common environment ───────────────────────────────────────────────
source "$SCRIPT_DIR/common_env.sh"

# ─── Set up data subset ─────────────────────────────────────────────────────
setup_data_subset "$EXPERIMENT_DIR"

# ─── Activate venv ───────────────────────────────────────────────────────────
echo "[run_experiment] Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# ─── Export experiment-specific vars ─────────────────────────────────────────
export ARCH_MODE="$ARCH_MODE"
export CKPT_DIR="$EXPERIMENT_DIR/checkpoints"
export RUN_ID="${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"

# ─── Training ────────────────────────────────────────────────────────────────
echo "=============================================="
echo " Training: $EXPERIMENT_NAME (arch=$ARCH_MODE)"
echo " Data subset: $DATA_PATH"
echo " Shard order: $SHARD_ORDER_FILE"
echo " Iterations: $ITERATIONS (warmdown at $WARMDOWN_ITERS)"
echo " Batch tokens: $TRAIN_BATCH_TOKENS"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "=============================================="

TRAIN_LOG="$EXPERIMENT_DIR/logs/train_${RUN_ID}.log"
cd "$EXPERIMENT_DIR/code"

# Use torchrun for proper distributed env init (even with 1 GPU)
torchrun --standalone --nproc_per_node=1 train_gdn_7k.py 2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT"
    echo "       Check log: $TRAIN_LOG"
    exit "$TRAIN_EXIT"
fi

echo ""
echo "[run_experiment] Training complete. Checkpoints in: $CKPT_DIR"

# ─── Find artifact for eval ─────────────────────────────────────────────────
# Prefer quantized .int6.ptz, fall back to .pt
ARTIFACT=""
for f in "$CKPT_DIR"/final_model_*.int6.ptz; do
    [ -f "$f" ] && ARTIFACT="$f" && break
done
if [ -z "$ARTIFACT" ]; then
    for f in "$CKPT_DIR"/final_model_*.pt; do
        [ -f "$f" ] && ARTIFACT="$f" && break
    done
fi

if [ -z "$ARTIFACT" ]; then
    echo "WARNING: No final model artifact found in $CKPT_DIR"
    echo "         Skipping eval_ttt."
    exit 0
fi

# ─── TTT Evaluation ─────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " Eval TTT: $EXPERIMENT_NAME"
echo " Artifact: $ARTIFACT"
echo "=============================================="

export ARTIFACT_PATH="$ARTIFACT"
EVAL_LOG="$EXPERIMENT_DIR/logs/eval_ttt_${RUN_ID}.log"

cd "$EXPERIMENT_DIR/code"
python eval_ttt.py 2>&1 | tee "$EVAL_LOG"
EVAL_EXIT=${PIPESTATUS[0]}

if [ "$EVAL_EXIT" -ne 0 ]; then
    echo "WARNING: Eval TTT failed with exit code $EVAL_EXIT"
    echo "         Check log: $EVAL_LOG"
fi

echo ""
echo "=============================================="
echo " Experiment complete: $EXPERIMENT_NAME"
echo " Train log: $TRAIN_LOG"
echo " Eval log:  $EVAL_LOG"
echo " Artifact:  $ARTIFACT"
echo "=============================================="
