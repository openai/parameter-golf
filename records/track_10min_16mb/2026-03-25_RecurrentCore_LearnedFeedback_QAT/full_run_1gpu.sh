#!/bin/bash
# Full 1-GPU run for 80 minutes to guesstimate final loss.
#
# On 1 GPU, grad_accum_steps=8 so each step is ~8x slower than 8-GPU.
# 80 minutes on 1 GPU ≈ 10 minutes on 8 GPUs in terms of training steps,
# giving a realistic estimate of the final BPB.
#
# Usage:
#   cd records/track_10min_16mb/2026-03-25_RecurrentCore_LearnedFeedback_QAT
#   bash full_run_1gpu.sh                          # default: learned feedback
#   bash full_run_1gpu.sh qat                      # QAT-only baseline
#   bash full_run_1gpu.sh fixed                    # fixed feedback
#   bash full_run_1gpu.sh learned                  # learned feedback
#   SEED=42 bash full_run_1gpu.sh learned          # custom seed
#   MINUTES=120 bash full_run_1gpu.sh learned      # longer run
#
# Prerequisites:
#   - 1 CUDA GPU (H100/H200/A100 recommended)
#   - Data downloaded: python data/cached_challenge_fineweb.py
#   - Dependencies: pip install sentencepiece numpy flash-attn

set -euo pipefail

VARIANT="${1:-learned}"
MINUTES="${MINUTES:-80}"
WALLCLOCK=$((MINUTES * 60))

export DATA_PATH="${DATA_PATH:-../../../data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-../../../data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-1337}"

# Full training config — matches the 8-GPU regime
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS="${WALLCLOCK}"
export VAL_LOSS_EVERY=2000
export TRAIN_LOG_EVERY=200
export WARMUP_STEPS=20
export WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export EVAL_STRIDE=64

# Model config
export NUM_STEM_LAYERS=3
export NUM_CORE_LAYERS=2
export NUM_TAIL_LAYERS=3
export NUM_PASSES=3
export CORE_QUANT_BITS=6
export CORE_QUANT_ENABLED=1
export BIGRAM_VOCAB_SIZE=1536
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="6,7"

# Optimizer config
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3

# Weight averaging
export SWA_ENABLED=1
export SWA_EVERY=50

# Late QAT
export LATE_QAT=1
export LATE_QAT_THRESHOLD=0.15

# TTT off for training run (enable separately for eval)
export TTT_ENABLED=0

echo "============================================================"
echo "  Full 1-GPU run: ${VARIANT} variant"
echo "  Wall clock: ${MINUTES} minutes (${WALLCLOCK}s)"
echo "  Seed: ${SEED}"
echo "  Data: ${DATA_PATH}"
echo "============================================================"

case "${VARIANT}" in
    qat)
        echo "Running: train_bestbase_recurrent_qat.py (QAT only, no feedback)"
        python train_bestbase_recurrent_qat.py \
            --ttt-regime tail_only
        ;;
    fixed)
        echo "Running: train_bestbase_recurrent_feedback_fixed.py (fixed diagonal feedback)"
        python train_bestbase_recurrent_feedback_fixed.py \
            --feedback-mode diagonal \
            --feedback-rank 2 \
            --ttt-regime tail_only
        ;;
    learned)
        echo "Running: train_bestbase_recurrent_feedback_learned.py (learned feedback)"
        python train_bestbase_recurrent_feedback_learned.py \
            --feedback-mode diagonal \
            --feedback-rank 2 \
            --ttt-regime tail_only
        ;;
    *)
        echo "Unknown variant: ${VARIANT}"
        echo "Usage: bash full_run_1gpu.sh [qat|fixed|learned]"
        exit 1
        ;;
esac

echo ""
echo "Run complete. Check logs/ for detailed output."
