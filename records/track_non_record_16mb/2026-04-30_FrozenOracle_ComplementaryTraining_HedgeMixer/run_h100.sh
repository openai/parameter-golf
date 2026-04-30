#!/bin/bash
# Parameter Golf — H100 Run Script (v4: Oracle + SGD TTT + LeakyReLU 0.75)
# Usage: bash run_h100.sh [seed]
# Requires: 8xH100 SXM, flash_attn_3, zstandard

set -e

SEED=${1:-1337}
echo "=== Parameter Golf: VR+GA+Oracle+SGD-TTT on PR #462 base ==="
echo "Seed: $SEED"

# Install deps
pip install -q zstandard

# Clone and setup
cd /workspace
git clone https://github.com/openai/parameter-golf.git 2>/dev/null || true
cd parameter-golf

# Download data (full 80 shards for competition run)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Copy our modified files
cp /workspace/train_gpt.py ./train_gpt.py
cp /workspace/build_ngram_oracle.py ./build_ngram_oracle.py

# === BUILD N-GRAM ORACLE (runs BEFORE the 10-min clock) ===
if [ ! -f /workspace/ngram_oracle.bin ]; then
    echo "Building n-gram oracle from training data..."
    python3 build_ngram_oracle.py \
        "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin" \
        /workspace/ngram_oracle.bin
    echo "Oracle built."
else
    echo "Using cached oracle at /workspace/ngram_oracle.bin"
fi

# Run training (600s) + TTT+eval (600s)
SEED=$SEED \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=8 \
MLP_HIDDEN=1792 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 \
VAL_BATCH_SIZE=524288 \
VOCAB_SIZE=1024 \
ITERATIONS=20000 \
WARMDOWN_ITERS=3500 \
WARMUP_STEPS=20 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=200 \
VALUE_RESIDUAL=1 \
GATED_ATTENTION=1 \
EMA_ENABLED=1 \
EMA_DECAY=0.9985 \
LATE_QAT=1 \
XSA_LAYERS=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LEAKY_RELU_SLOPE=0.75 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
TTT_ENABLED=1 \
TTT_OPTIMIZER=sgd \
TTT_LR=0.002 \
TTT_EPOCHS_PER_CHUNK=20 \
TTT_CHUNK_TOKENS=131072 \
TTT_FREEZE_BLOCKS=9 \
TTT_TIME_BUDGET=450 \
EVAL_STRIDE=64 \
BIGRAM_BUCKETS=8192 \
BIGRAM_EMBED_DIM=128 \
CROWN_LAMBDA=0.0001 \
HEDGE_ENABLED=1 \
HEDGE_ETA=0.01 \
HEDGE_TRIGRAM_BUCKETS=8192 \
HEDGE_CACHE_GAMMA=0.999 \
NGRAM_ORACLE_PATH=/workspace/ngram_oracle.bin \
RUN_ID="oracle_sgdttt_hedge_seed${SEED}" \
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo "=== Run complete for seed $SEED ==="
