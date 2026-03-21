#!/bin/bash
# === Parameter Golf: Competitive Entry (SOTA stack) ===
# Based on thwu1's #1 submission (1.1428 bpb)
# Techniques: 10L + BigramHash(10240) + SmearGate + Int5/Int6 mixed quant
#             + 3x MLP + OrthoInit + SWA(0.4) + WD=0.04 + sliding eval
#             + zstd-22 + magnitude pruning
#
# Requirements: 8x H100 SXM (or adjust WORLD_SIZE)
# Expected: ~1.14 bpb in 10 minutes
#
# Usage:
#   # 8x H100 (full competitive run)
#   bash run_competitive.sh
#
#   # 1x H100 (pilot test)
#   bash run_competitive.sh --pilot

set -e

PILOT=0
if [[ "$1" == "--pilot" ]]; then
    PILOT=1
fi

cd /workspace/parameter-golf

# Install zstandard for better compression
pip install zstandard 2>/dev/null || true

# Download dataset if not present
if [ ! -f data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin ]; then
    echo "=== Downloading dataset ==="
    export HF_TOKEN="${HF_TOKEN:-}"
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

echo "Train shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)"
echo "Val shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)"

if [ "$PILOT" -eq 1 ]; then
    echo ""
    echo "=== PILOT RUN: 1x H100, 10 min ==="
    echo "Start: $(date)"
    NPROC=1
else
    echo ""
    echo "=== COMPETITIVE RUN: 8x H100, 10 min ==="
    echo "Start: $(date)"
    NPROC=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected GPUs: $NPROC"
fi

RUN_ID="competitive_$(date +%Y%m%d_%H%M%S)" \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
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
torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /workspace/competitive_log.txt

echo ""
echo "=== RESULTS ==="
grep -E 'val_bpb|final_int8|submission|model_params|swa:' /workspace/competitive_log.txt | tail -20
echo ""
echo "Target: val_bpb < 1.1428 (current SOTA)"
echo "Done: $(date)"
