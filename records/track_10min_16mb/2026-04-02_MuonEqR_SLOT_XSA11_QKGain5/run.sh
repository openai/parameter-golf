#!/bin/bash
set -euo pipefail

SEED=${SEED:-1337}
NPROC=${NPROC:-8}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== MuonEq-R + SLOT + XSA-all + QK-Gain 5.0 ==="
echo "Seed: $SEED | GPUs: $NPROC"

export NUM_LAYERS=11
export BIGRAM_VOCAB_SIZE=1536
export XSA_LAST_N=11
export QK_GAIN_INIT=5.0
export EMA_ENABLED=1
export EMA_DECAY=0.997
export SWA_ENABLED=1
export SWA_EVERY=50
export ROPE_DIMS=16
export LN_SCALE=1
export LATE_QAT_THRESHOLD=0.15
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS=9,10
export TTT_ENABLED=1
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0
export SLOT_ENABLED=1
export SLOT_STEPS=8
export SLOT_LR=0.005
export MUON_WD=0.04
export ADAM_WD=0.04
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3500
export ITERATIONS=9000
export MAX_WALLCLOCK_SECONDS=600
export EVAL_STRIDE=64
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=200
export SEED=$SEED

torchrun --standalone --nproc_per_node=$NPROC "$SCRIPT_DIR/train_gpt.py"
