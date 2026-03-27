#!/bin/bash
set -euo pipefail

PYTHON="/home/nesta/parameter-golf/.venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Smoke test: same params as full run, 50 iterations, torch.compile ENABLED ==="

PYTHONUNBUFFERED=1 \
DATA_PATH="../../../data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="../../../data/tokenizers/fineweb_1024_bpe.model" \
SEED=1337 \
ITERATIONS=50 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=25 \
TRAIN_LOG_EVERY=10 \
WARMUP_STEPS=5 \
WARMDOWN_ITERS=10 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=0 \
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
MUON_MOMENTUM_WARMUP_STEPS=5 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
GRAD_CLIP_NORM=0.3 \
SWA_ENABLED=1 \
SWA_EVERY=10 \
LATE_QAT=0 \
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
echo "=== Smoke test complete ==="
