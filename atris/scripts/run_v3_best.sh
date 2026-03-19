#!/bin/bash
# Atris v3 BEST: Stack ALL proven improvements
#
# Combines:
#   1. 10 layers (validated by nanlliu, +0.01 BPB)
#   2. Lower LR 0.02/0.02/0.03 (consensus)
#   3. MLP 3x wider (validated by jfprincz, +0.02 BPB)
#   4. INT6 middle layers (saves ~1.6MB for wider MLP)
#   5. Sliding window eval stride=64 (+0.03 BPB FREE)
#   6. Muon momentum 0.99 (validated by yesbhautik)
#
# Expected: ~1.16 BPB on standard train data
#
# For SP-4096 variant (larger vocab, better bytes/token):
#   VARIANT=sp4096 bash atris/scripts/run_v3_best.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

VARIANT=${VARIANT:-sp1024}
NPROC=${NPROC:-8}
WALLCLOCK=${WALLCLOCK:-600}

# Download dataset for this variant if needed
if [ ! -d "./data/datasets/fineweb10B_${VARIANT}" ]; then
    echo "Downloading dataset variant: $VARIANT ..."
    python3 data/cached_challenge_fineweb.py --variant "$VARIANT"
fi

# Determine vocab size and tokenizer from variant
if [ "$VARIANT" = "sp4096" ]; then
    VOCAB=4096
    TOK_PATH="./data/tokenizers/fineweb_4096_bpe.model"
elif [ "$VARIANT" = "sp1024" ]; then
    VOCAB=1024
    TOK_PATH="./data/tokenizers/fineweb_1024_bpe.model"
else
    echo "Unknown variant: $VARIANT"
    exit 1
fi

# Copy our modified train_gpt.py (with sliding window + INT6 + MLP 3x)
if [ ! -f train_gpt.py.orig ]; then
    cp train_gpt.py train_gpt.py.orig
fi
cp atris/experiments/v1_train_gpt.py train_gpt.py

echo "================================================"
echo "  ATRIS v3 BEST: All proven improvements"
echo "  Variant: $VARIANT (vocab=$VOCAB)"
echo "  GPUs: $NPROC | Wallclock: ${WALLCLOCK}s"
echo "  Sliding window: stride=64"
echo "================================================"

NCCL_IB_DISABLE=1 \
RUN_ID="atris_v3_${VARIANT}_$(date +%s)" \
DATA_PATH="./data/datasets/fineweb10B_${VARIANT}/" \
TOKENIZER_PATH="$TOK_PATH" \
VOCAB_SIZE=$VOCAB \
NUM_LAYERS=10 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MLP_MULT=3 \
EVAL_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=$WALLCLOCK \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee atris/logs/v3_${VARIANT}_run.log

echo ""
echo "================================================"
echo "  Run complete. Key metrics:"
grep -E "(final_int8_zlib_roundtrip|submission size)" atris/logs/v3_${VARIANT}_run.log || true
echo "================================================"
echo ""
echo "Next: If artifact > 16MB, try reducing NUM_LAYERS or MODEL_DIM"
echo "      If BPB looks good, run 5 seeds for statistical significance"
