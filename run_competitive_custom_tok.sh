#!/bin/bash
# === Parameter Golf: Competitive Entry with Custom Tokenizer ===
# Same SOTA stack as run_competitive.sh but uses a custom-trained tokenizer.
#
# WORKFLOW:
# 1. Train tokenizer: python3 ../trainer-tokenizer/train_tokenizer.py --vocab-size 1024 --model-type unigram
# 2. Export shards:   python3 ../trainer-tokenizer/train_tokenizer.py --vocab-size 1024 --model-type unigram --export-shards
# 3. Run this script with the paths
#
# Usage:
#   CUSTOM_TOKENIZER=./tokenizers/spm_unigram_1024.model \
#   CUSTOM_DATA=./tokenizers/shards_unigram_1024 \
#   CUSTOM_VOCAB=1024 \
#   bash run_competitive_custom_tok.sh [--pilot]

set -e

PILOT=0
if [[ "$1" == "--pilot" ]]; then
    PILOT=1
fi

# Custom tokenizer paths (must be set)
CUSTOM_TOKENIZER="${CUSTOM_TOKENIZER:?Set CUSTOM_TOKENIZER to your .model file}"
CUSTOM_DATA="${CUSTOM_DATA:?Set CUSTOM_DATA to your shard directory}"
CUSTOM_VOCAB="${CUSTOM_VOCAB:-1024}"

cd /workspace/parameter-golf

pip install zstandard 2>/dev/null || true

echo "=== Custom Tokenizer Run ==="
echo "Tokenizer: $CUSTOM_TOKENIZER"
echo "Data: $CUSTOM_DATA"
echo "Vocab: $CUSTOM_VOCAB"
echo "Train shards: $(ls ${CUSTOM_DATA}/fineweb_train_*.bin 2>/dev/null | wc -l)"
echo "Val shards: $(ls ${CUSTOM_DATA}/fineweb_val_*.bin 2>/dev/null | wc -l)"

if [ "$PILOT" -eq 1 ]; then
    NPROC=1
    echo "Mode: PILOT (1x GPU)"
else
    NPROC=$(nvidia-smi --list-gpus | wc -l)
    echo "Mode: COMPETITIVE ($NPROC GPUs)"
fi

echo "Start: $(date)"

RUN_ID="custom_tok_$(date +%Y%m%d_%H%M%S)" \
DATA_PATH="$CUSTOM_DATA" \
TOKENIZER_PATH="$CUSTOM_TOKENIZER" \
VOCAB_SIZE="$CUSTOM_VOCAB" \
SEED=42 \
NUM_LAYERS=10 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3.0 \
TIE_EMBEDDINGS=1 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
WEIGHT_DECAY=0.04 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
GRAD_CLIP_NORM=0.3 \
BIGRAM_VOCAB_SIZE=10240 \
BIGRAM_DIM=128 \
SWA_ENABLED=1 \
SWA_START_FRAC=0.4 \
SWA_EVERY=50 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=32 \
torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /workspace/custom_tok_log.txt

echo ""
echo "=== RESULTS ==="
grep -E 'val_bpb|final_int8|submission|model_params|swa:' /workspace/custom_tok_log.txt | tail -20
echo ""
echo "Done: $(date)"
