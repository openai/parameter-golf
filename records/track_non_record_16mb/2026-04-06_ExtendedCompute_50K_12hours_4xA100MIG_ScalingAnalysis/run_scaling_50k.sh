#!/bin/bash
set -e

cd /projects/jundhu-v2/jundhu_projects/parameter_golf/parameter_golf_exp

# 50K scaling run — intermediate val_bpb logged every 2.5K steps for scaling curve
# Warmdown and momentum warmup scaled proportionally (~39% and ~16.7%)
# 50K * 0.39 = 19500, 50K * 0.167 = 8350

echo "=========================================="
echo "Scaling run: 50K steps, seed=2024"
echo "Started at: $(date)"
echo "=========================================="

env \
    RUN_ID=train_step50k_seed2024 \
    NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
    EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
    ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
    TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
    MUON_WD=0.04 ADAM_WD=0.04 \
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=8350 WARMDOWN_ITERS=19500 \
    ITERATIONS=50000 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
    VAL_LOSS_EVERY=2500 \
    SEED=2024 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --standalone --nproc_per_node=4 train_gpt.py \
    2>&1 | tee logs/train_step50k_seed2024.txt

echo "=========================================="
echo "50K run finished at: $(date)"
echo "=========================================="
