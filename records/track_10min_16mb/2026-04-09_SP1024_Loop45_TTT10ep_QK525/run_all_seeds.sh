#!/bin/bash
# Run 009: Apply PR #1487 TTT hyperparameter tuning to our SP1024 + Looping architecture
# Hypothesis: TTT 10ep + lr=0.00045 + freeze=1 + QK=5.25 will gain ~0.008 BPB over Run 007/008
# Expected: val_bpb ~1.066 (vs 1.0739 baseline)

set -e

# Core architecture (same as Run 007/008)
export VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4.0
export NUM_LOOPS=2 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.5
export PARALLEL_START_LAYER=7

# TTT hyperparameters (PR #1487 tuning)
export PREQUANT_TTT_ENABLED=1
export PREQUANT_TTT_LR=0.00045        # was 0.0005
export PREQUANT_TTT_EPOCHS=10          # was 6
export PREQUANT_TTT_FREEZE_BLOCKS=1    # was 2
export PREQUANT_TTT_BATCH_SEQS=32
export PREQUANT_TTT_GRAD_CLIP=1.0
export PREQUANT_TTT_COSINE_DECAY=1

# QK-Gain (PR #1487 tuning)
export QK_GAIN_INIT=5.25               # was 5.0

# Other settings (same as Run 007/008)
export EMA_DECAY=0.9965
export EMBED_BITS=8 MATRIX_BITS=6 COMPRESSOR=brotli GPTQ_ENABLED=1
export SLIDING_WINDOW_ENABLED=1 ETLB_ENABLED=1
export TRAIN_SEQ_LEN=2048 MAX_WALLCLOCK_SECONDS=600 WARMDOWN_FRAC=0.667 WARMUP_STEPS=20
export TRAIN_BATCH_TOKENS=786432
export MIN_LR=0.0 EMBED_LR=0.6 HEAD_LR=0.008 TIED_EMBED_LR=0.03 MATRIX_LR=0.04 SCALAR_LR=0.02

# Run 3 seeds for statistical significance
for SEED in 314 42 999; do
    echo "=== Run 009: Seed $SEED ==="
    echo "TTT: 10ep, lr=0.00045, freeze=1 | QK-Gain: 5.25"
    export SEED=$SEED
    torchrun --nproc_per_node=8 records/track_10min_16mb/2026-04-09_SP1024_Loop45_TTT10ep_QK525/train_gpt.py
done
