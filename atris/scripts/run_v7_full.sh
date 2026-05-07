#!/bin/bash
# Atris v7 FULL: All techniques from the top 3 leaderboard entries
#
# Stacks everything:
#   v1: 10 layers, lower LR 0.02
#   v2: sliding window eval stride=64
#   v5: QAT (INT8 fake quant during training)
#   v6: SWA, weight decay 0.04, gradient clipping 0.3
#   v7: BigramHash(4096) + SmearGate
#
# Size-safe configs (verified to fit in 16MB):
#   10L MLP2x 512d = ~13.3MB + BigramHash ~0.5MB = ~13.8MB ✅
#   9L MLP3x 512d  = ~15.3MB + BigramHash ~0.5MB = ~15.8MB ✅ (tight!)

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ ! -f train_gpt.py.orig ]; then cp train_gpt.py train_gpt.py.orig; fi
cp atris/experiments/v1_train_gpt.py train_gpt.py

NPROC=${NPROC:-8}
WALLCLOCK=${WALLCLOCK:-600}
VARIANT=${VARIANT:-sp1024}

if [ "$VARIANT" = "sp4096" ]; then
    VOCAB=4096; TOK="./data/tokenizers/fineweb_4096_bpe.model"
else
    VOCAB=1024; TOK="./data/tokenizers/fineweb_1024_bpe.model"
fi

echo "================================================"
echo "  ATRIS v7 FULL: Complete technique stack"
echo "  10L + SWA + WD + BigramHash + SmearGate"
echo "  GPUs: $NPROC | Variant: $VARIANT"
echo "================================================"

NCCL_IB_DISABLE=1 \
RUN_ID="atris_v7_$(date +%s)" \
DATA_PATH="./data/datasets/fineweb10B_${VARIANT}/" \
TOKENIZER_PATH="$TOK" \
VOCAB_SIZE=$VOCAB \
NUM_LAYERS=10 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_WEIGHT_DECAY=0.04 \
ADAM_WEIGHT_DECAY=0.01 \
GRAD_CLIP_NORM=0.3 \
QAT_BITS=8 \
EVAL_STRIDE=64 \
SWA_EVERY=50 \
SWA_START_FRAC=0.4 \
BIGRAM_BUCKETS=4096 \
BIGRAM_DIM=128 \
SMEAR_GATE=1 \
MAX_WALLCLOCK_SECONDS=$WALLCLOCK \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee atris/logs/v7_full_run.log

echo ""
echo "================================================"
grep -E "(final_int8_zlib_roundtrip|submission size|model_params|swa:)" atris/logs/v7_full_run.log || true
echo "================================================"
