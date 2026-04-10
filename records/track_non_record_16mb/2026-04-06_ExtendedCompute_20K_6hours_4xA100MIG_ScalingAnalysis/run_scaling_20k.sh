#!/bin/bash
set -e

cd "$(dirname "$0")"

# 20k scaling run — intermediate val_bpb logged every 2K steps for scaling curve
# Warmdown and momentum warmup scaled proportionally (~39% and ~16.7%)
# 20k * 0.39 = 7800, 20k * 0.167 = 3340

echo "=========================================="
echo "Scaling run: 20k steps, seed=2024"
echo "Started at: $(date)"
echo "=========================================="

env \
    RUN_ID=train_step20k_seed2024 \
    NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
    EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
    ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
    TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
    MUON_WD=0.04 ADAM_WD=0.04 \
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=3340 WARMDOWN_ITERS=7800 \
    ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
    VAL_LOSS_EVERY=2000 \
    SEED=2024 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --standalone --nproc_per_node=4 train_gpt.py \
    2>&1 | tee logs/train_step20k_seed2024.txt

echo "=========================================="
echo "20k run finished at: $(date)"
echo "=========================================="
