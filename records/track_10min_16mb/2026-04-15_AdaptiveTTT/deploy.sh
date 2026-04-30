#!/bin/bash
# Parameter Golf $25 Budget Deployment Script
# =============================================
# 3-run plan: reproduce SOTA, test adaptive TTT, confirm with 2nd seed
#
# Usage on RunPod 8xH100:
#   bash experiments/deploy.sh reproduce_sota 42
#   bash experiments/deploy.sh adaptive_ttt 42
#   bash experiments/deploy.sh adaptive_ttt 1337   # 2nd seed if Run 2 improved
#   bash experiments/deploy.sh adaptive_safe 42     # fallback if Run 2 regressed
#
# Budget: $25 = 1 hour on 8xH100 = ~3 runs of 20 min each

set -e

EXPERIMENT="${1:-reproduce_sota}"
SEED="${2:-42}"
NPROC="${3:-8}"

echo "============================================="
echo "Parameter Golf: ${EXPERIMENT} (seed=${SEED}, ${NPROC} GPUs)"
echo "============================================="

cd /workspace/parameter-golf 2>/dev/null || cd ~/parameter-golf 2>/dev/null || {
    echo "ERROR: parameter-golf directory not found"
    echo "Clone with: git clone https://github.com/openai/parameter-golf.git"
    exit 1
}

# Ensure SP8192 dataset + tokenizer are available
if [ ! -d "data/datasets/fineweb10B_sp8192" ]; then
    echo "Downloading SP8192 dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 10
fi

# Dependencies
pip install brotli sentencepiece -q 2>/dev/null

echo ""
echo "Starting: ${EXPERIMENT} (seed=${SEED})"
echo "============================================="

# Common env vars matching SOTA #1 defaults
COMMON="VOCAB_SIZE=8192 TTT_ENABLED=1 COMPRESSOR=brotli"

EXTRA_ENV=""
SCRIPT="experiments/train_gpt_v1.py"
case "$EXPERIMENT" in
    reproduce_sota)
        # Run 1: Exact SOTA reproduction (adaptive OFF, SGD TTT, no LR floor)
        EXTRA_ENV="TTT_ADAPTIVE=0 TTT_USE_ADAM=0 TTT_MIN_LR_FRAC=0"
        ;;
    adaptive_ttt)
        # Run 2: Novel adaptive TTT (more epochs on hard chunks, fewer on easy)
        EXTRA_ENV="TTT_ADAPTIVE=1 TTT_USE_ADAM=0 TTT_MAX_EPOCHS=5 TTT_MIN_EPOCHS=1 TTT_ADAPT_EMA=0.3 TTT_MIN_LR_FRAC=0.1"
        ;;
    adaptive_safe)
        # Fallback: Just TTT LR floor (no adaptive epochs)
        EXTRA_ENV="TTT_ADAPTIVE=0 TTT_USE_ADAM=0 TTT_MIN_LR_FRAC=0.1"
        ;;
    adaptive_aggressive)
        # More aggressive adaptive: wider epoch range, bigger EMA weight
        EXTRA_ENV="TTT_ADAPTIVE=1 TTT_USE_ADAM=0 TTT_MAX_EPOCHS=6 TTT_MIN_EPOCHS=1 TTT_ADAPT_EMA=0.5 TTT_MIN_LR_FRAC=0.15"
        ;;
    *)
        echo "Unknown experiment: ${EXPERIMENT}"
        echo "Known: reproduce_sota, adaptive_ttt, adaptive_safe, adaptive_aggressive"
        exit 1
        ;;
esac

LOG_FILE="/workspace/results_${EXPERIMENT}_s${SEED}.log"

echo "Script: ${SCRIPT}"
echo "Log: ${LOG_FILE}"

# Launch training on all GPUs
env $COMMON $EXTRA_ENV SEED=$SEED RUN_ID="exp_${EXPERIMENT}_s${SEED}" \
    torchrun --standalone --nproc_per_node=${NPROC} ${SCRIPT} 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================="
echo "RESULTS: ${EXPERIMENT} (seed=${SEED})"
echo "============================================="
grep -E "val_bpb|quantized|ttt:|adaptive|chunk=|peak memory|submission" "$LOG_FILE" | tail -30
echo ""
echo "Full log: ${LOG_FILE}"
