#!/bin/bash
# Full reproduction script for RunPod 8xH100 SXM
# Downloads data, trains, quantizes, evaluates

set -e

# Install dependencies
pip install sentencepiece huggingface-hub datasets tqdm brotli zstandard wandb flash-attn

# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# Train + quantize + eval
SEED=1337 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
NUM_LAYERS=14 BIGRAM_VOCAB_SIZE=8192 BIGRAM_DIM=64 \
MUON_WD=0.09 ADAM_WD=0.02 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 \
EVAL_STRIDE=76 MLP_ACTIVATION=leaky2 \
TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 \
ROPE_BASE=50000 SWA_ENABLED=0 \
GPTQ_ENABLED=1 GPTQ_SAMPLES=256 QEP_ENABLED=1 \
WANDB_RUN_NAME=submission_14L_QEP_TTT \
torchrun --standalone --nproc_per_node=8 train_gpt.py
