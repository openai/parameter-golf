#!/bin/bash
# Run the best configuration found from 120+ experiments
# Best result: 1.5207 BPB on RTX 4000 Ada (276 steps in 600s)
#
# Usage: bash run_best.sh
# Requires: GPU with PyTorch, same environment as train_gpt.py

cd "$(dirname "$0")"

env \
  TIDAL_LR=1 \
  LOGIT_SOFTCAP=15.0 \
  ROPE_BASE=5000 \
  PARALLEL_BLOCK=1 \
  MLP_ACT=silu2 \
  HEAD_DIVERSITY=1e-4 \
  EMBED_LR=0.8 \
  MATRIX_LR=0.11 \
  ENCODER_LAYERS=0 \
  NUM_KV_HEADS=2 \
  TIE_EMBEDDINGS=0 \
  python train_gpt_focal_fixed.py
