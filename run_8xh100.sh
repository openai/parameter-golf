#!/bin/bash
# Parameter Golf - 8xH100 Validation Script
# Based on PR #198 (SOTA 1.1318) + 12L + Int5 MLP innovation
set -e

echo "=== Parameter Golf 8xH100 ==="
echo "Start: $(date)"

cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/alertcat/parameter-golf.git
fi
cd parameter-golf

# Download data if not present
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo "=== Downloading data ==="
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
fi

echo "=== Environment ==="
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
N_GPUS=$(python3 -c 'import torch; print(torch.cuda.device_count())')

# ============================================================
# CONFIG: 12 layers + Int5 MLP (our innovation over PR #198)
# PR #198 = 11L int6 = 1.1318
# Ours = 12L + int5 MLP = saves ~1.8MB, funds 12th layer
# ============================================================
export NUM_LAYERS=12
export INT5_MLP=1
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export VOCAB_SIZE=1024
export TIE_EMBEDDINGS=1
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export ITERATIONS=9000
export MAX_WALLCLOCK_SECONDS=600
export WARMDOWN_ITERS=3000
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04
export ADAM_WD=0.04
export QAT_ENABLED=1
export SMEAR_GATE=1
export BIGRAM_HASH=1
export BIGRAM_VOCAB_SIZE=2048
export BIGRAM_DIM=128
export SWA_ENABLED=1
export SWA_EVERY=200
export EVAL_STRIDE=64
export EVAL_BATCH_SEQS=32
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
export GRAD_CLIP_NORM=0.3

for SEED_VAL in 1337 42 2024; do
    echo ""
    echo "======================================================================"
    echo "=== SEED $SEED_VAL | $(date) ==="
    echo "======================================================================"
    export SEED=$SEED_VAL
    export RUN_ID="12L_int5_s${SEED_VAL}"
    torchrun --standalone --nproc_per_node=$N_GPUS train_gpt.py 2>&1 | tee "run_seed${SEED_VAL}.txt"
    echo "--- Seed $SEED_VAL complete ---"
done

echo ""
echo "======================================================================"
echo "=== ALL SEEDS COMPLETE ==="
echo "======================================================================"
for f in run_seed*.txt; do
    echo "--- $f ---"
    grep -E "final_int6.*val_bpb|Total submission size|sliding.*val_bpb" "$f" | tail -3
done
echo "End: $(date)"
