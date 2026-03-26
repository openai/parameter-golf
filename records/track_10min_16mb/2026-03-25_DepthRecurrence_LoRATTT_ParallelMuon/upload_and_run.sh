#!/bin/bash
# Upload this script and train_gpt.py to your RunPod, then run from /workspace/parameter-golf/

set -e

# ============================================================
# STEP 1: Copy the submission script into position
# ============================================================
cp /workspace/parameter-golf/submission/train_gpt.py /workspace/parameter-golf/train_gpt_sub.py

cd /workspace/parameter-golf

# ============================================================
# STEP 2: Make sure data is downloaded
# ============================================================
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
fi

# ============================================================
# STEP 3: Quick 1xH100 smoke test (disable TTT, ~10 min)
# ============================================================
echo ""
echo "=========================================="
echo "RUNNING: 1xH100 smoke test (no TTT)"
echo "=========================================="

RUN_ID=smoke_depth_recurrence_v1 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
DEPTH_RECURRENCE=4,5 \
EMA_ENABLED=1 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
GRAD_CLIP_NORM=0.3 \
TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt_sub.py
