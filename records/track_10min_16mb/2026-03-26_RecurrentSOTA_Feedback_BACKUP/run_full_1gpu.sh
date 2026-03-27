#!/bin/bash
set -euo pipefail

PYTHON="/home/nesta/parameter-golf/.venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MINUTES="${MINUTES:-80}"
WALLCLOCK=$((MINUTES * 60))
SEED="${SEED:-1337}"

echo "============================================================"
echo "  Full 1-GPU run: RecurrentSOTA + Learned Feedback"
echo "  Wall clock: ${MINUTES} minutes (${WALLCLOCK}s)"
echo "  Seed: ${SEED}"
echo "============================================================"

PYTHONUNBUFFERED=1 \
DATA_PATH="../../../data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="../../../data/tokenizers/fineweb_1024_bpe.model" \
SEED="${SEED}" \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=200 \
WARMUP_STEPS=20 \
WARMDOWN_ITERS=3500 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS="9,10" \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
GRAD_CLIP_NORM=0.3 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
LATE_QAT=1 \
LATE_QAT_THRESHOLD=0.15 \
TTT_ENABLED=0 \
CORE_START=3 \
CORE_END=8 \
NUM_PASSES=2 \
CORE_QUANT_ENABLED=0 \
$PYTHON train_gpt_recurrent.py \
    --feedback-mode diagonal --feedback-rank 2 \
    --residual-scale-init 0.5 \
    --jacobian-proxy-weight 0.01

echo ""
echo "Run complete."
