#!/bin/bash
# Atris v4 SHARED: Weight sharing + wider model
#
# 4 unique blocks × 3 repeats = 12 effective layers
# Freed params → MODEL_DIM=768 (50% wider)
# + all v3 improvements (sliding window, MLP 3x, INT6, lower LR)
#
# The insight: shared blocks produce identical weight patterns in the state dict.
# zlib/zstd compresses repeated patterns nearly to zero. Double compression win.

set -euo pipefail

cd "$(dirname "$0")/../.."

if [ ! -f train_gpt.py.orig ]; then
    cp train_gpt.py train_gpt.py.orig
fi
cp atris/experiments/v1_train_gpt.py train_gpt.py

NPROC=${NPROC:-8}
WALLCLOCK=${WALLCLOCK:-600}
VARIANT=${VARIANT:-sp1024}

if [ "$VARIANT" = "sp4096" ]; then
    VOCAB=4096; TOK_PATH="./data/tokenizers/fineweb_4096_bpe.model"
else
    VOCAB=1024; TOK_PATH="./data/tokenizers/fineweb_1024_bpe.model"
fi

echo "================================================"
echo "  ATRIS v4 SHARED: Weight sharing + wider model"
echo "  4 unique blocks × 3 = 12 effective layers"
echo "  MODEL_DIM=768 | MLP 3x | Sliding window"
echo "  Variant: $VARIANT | GPUs: $NPROC"
echo "================================================"

NCCL_IB_DISABLE=1 \
RUN_ID="atris_v4_shared_$(date +%s)" \
DATA_PATH="./data/datasets/fineweb10B_${VARIANT}/" \
TOKENIZER_PATH="$TOK_PATH" \
VOCAB_SIZE=$VOCAB \
NUM_LAYERS=12 \
NUM_UNIQUE_BLOCKS=4 \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
EVAL_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=$WALLCLOCK \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee atris/logs/v4_shared_run.log

echo ""
echo "================================================"
grep -E "(final_int8_zlib_roundtrip|submission size|model_params)" atris/logs/v4_shared_run.log || true
echo "================================================"
echo ""
echo "NOTE: If artifact > 16MB, reduce MLP_MULT to 2 or MODEL_DIM to 640"
