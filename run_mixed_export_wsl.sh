#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/wrc02/Desktop/Projects/NanoGPT-Challenge/repo
mkdir -p logs
env \
  PYTHONUNBUFFERED=1 \
  ATTN_EVERY=2 \
  NUM_LAYERS=12 \
  MODEL_DIM=512 \
  NHEADS=16 \
  D_STATE=16 \
  D_CONV=4 \
  CHUNK_SIZE=64 \
  EXPAND=2 \
  ATTN_NHEADS=8 \
  ATTN_KV_HEADS=2 \
  VOCAB_SIZE=1024 \
  TRAIN_SEQ_LEN=1024 \
  TRAIN_BATCH_TOKENS=65536 \
  ITERATIONS=800 \
  WARMUP_STEPS=5 \
  VAL_LOSS_EVERY=400 \
  TRAIN_LOG_EVERY=100 \
  WARMDOWN_ITERS=100 \
  MAX_WALLCLOCK_SECONDS=0 \
  CGGR_WARMUP=200 \
  CGGR_RATIO=0.5 \
  MATRIX_LR=0.025 \
  SCALAR_LR=0.04 \
  TIED_EMBED_LR=0.05 \
  SEED=1337 \
  RUN_ID=mixed_export_outproj_wsl_20260320_195359 \
  python3 -u train_gpt.py