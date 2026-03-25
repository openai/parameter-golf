#!/bin/bash
# Quick 1-GPU smoke test to validate code correctness.
# Runs ~100 steps with small settings — NOT for competitive BPB.
#
# Usage:
#   cd records/track_10min_16mb/2026-03-25_RecurrentCore_LearnedFeedback_QAT
#   bash smoke_test.sh
#
# Prerequisites:
#   - 1+ CUDA GPU
#   - Data downloaded: python data/cached_challenge_fineweb.py
#   - Dependencies: pip install sentencepiece numpy flash-attn

set -euo pipefail

export DATA_PATH="../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../data/tokenizers/fineweb_1024_bpe.model"

# Minimal settings for a quick correctness check
export ITERATIONS=100
export MAX_WALLCLOCK_SECONDS=120
export VAL_LOSS_EVERY=50
export TRAIN_LOG_EVERY=10
export WARMUP_STEPS=5
export WARMDOWN_ITERS=30
export TRAIN_BATCH_TOKENS=131072
export TTT_ENABLED=0

# Recurrence config
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
export SWA_ENABLED=0

echo "=== Smoke test: train_bestbase_recurrent_qat.py (QAT only) ==="
python train_bestbase_recurrent_qat.py --ttt-regime tail_only 2>&1 | tail -20

echo ""
echo "=== Smoke test: train_bestbase_recurrent_feedback_fixed.py (fixed feedback) ==="
python train_bestbase_recurrent_feedback_fixed.py \
    --feedback-mode diagonal --feedback-rank 2 --ttt-regime tail_only 2>&1 | tail -20

echo ""
echo "=== Smoke test: train_bestbase_recurrent_feedback_learned.py (learned feedback) ==="
python train_bestbase_recurrent_feedback_learned.py \
    --feedback-mode diagonal --feedback-rank 2 --ttt-regime tail_only 2>&1 | tail -20

echo ""
echo "All smoke tests passed!"
