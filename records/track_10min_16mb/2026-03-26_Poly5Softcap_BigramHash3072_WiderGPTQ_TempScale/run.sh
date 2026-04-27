#!/bin/bash
# Run training + evaluation with all improvements enabled
# Requires: 8xH100 80GB SXM

set -euo pipefail

# Download data if needed
python3 data/cached_challenge_fineweb.py --variant sp1024

# --- Training configuration ---
export SEED=${SEED:-1337}
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=128
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export TIE_EMBEDDINGS=1
export TIED_EMBED_LR=0.035
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.5
export WARMDOWN_ITERS=3500
export WARMUP_STEPS=20
export LATE_QAT_THRESHOLD=0.15
export SWA_ENABLED=1
export SWA_EVERY=50
export EVAL_STRIDE=64
export MAX_WALLCLOCK_SECONDS=600.0
export LOGIT_SOFTCAP=30.0
export SOFTCAP_TYPE=poly
export Z_LOSS_WEIGHT=1e-4
export EVAL_TEMPERATURE=0.95

# --- Legal TTT ---
export TTT_ENABLED=1
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

# --- Run ---
torchrun --nproc_per_node=8 \
    records/track_10min_16mb/2026-03-26_Poly5Softcap_BigramHash3072_WiderGPTQ_TempScale/train_gpt.py \
    2>&1 | tee "train_seed${SEED}.log"
