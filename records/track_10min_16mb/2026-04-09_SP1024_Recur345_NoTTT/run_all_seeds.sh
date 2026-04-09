#!/bin/bash
# Run 010: Track A Baseline — Depth Recurrence (no TTT, no looping)
# Hypothesis: 3-layer depth recurrence (L3-5) beats our 2-loop on L4-5
# Expected: ~1.08-1.09 BPB (architecture gain, no TTT)

set -e

# Core architecture
export VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4.0

# Depth recurrence (replaces looping)
export DEPTH_RECUR_ENABLED=1
export DEPTH_RECUR_LAYERS="3,4,5"
export DEPTH_RECUR_START_STEP=2000

# Parallel residuals
export PARALLEL_START_LAYER=7

# NO TTT (Track A - no adaptation)
export PREQUANT_TTT_ENABLED=0

# Hyperparameters (PR #1487 tuning)
export QK_GAIN_INIT=5.25
export EMA_DECAY=0.9965
export ADAM_WD=0.095
export MUON_WD=0.095
export MATRIX_LR=0.022
export WARMDOWN_FRAC=0.72

# Quantization and eval
export EMBED_BITS=8 MATRIX_BITS=6 COMPRESSOR=brotli GPTQ_ENABLED=1
export SLIDING_WINDOW_ENABLED=1 ETLB_ENABLED=1
export TRAIN_SEQ_LEN=2048 MAX_WALLCLOCK_SECONDS=590 WARMUP_STEPS=20
export TRAIN_BATCH_TOKENS=786432
export MIN_LR=0.0 EMBED_LR=0.6 HEAD_LR=0.008 TIED_EMBED_LR=0.03 SCALAR_LR=0.02

# Run 3 seeds for statistical significance
for SEED in 314 42 999; do
    echo "=== Run 010: Seed $SEED ==="
    echo "Depth Recurrence: L3-5 | No TTT | QK=5.25 | WD=0.095"
    export SEED=$SEED
    torchrun --nproc_per_node=8 records/track_10min_16mb/2026-04-09_SP1024_Recur345_NoTTT/train_gpt.py
done
