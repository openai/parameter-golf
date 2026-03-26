#!/bin/bash
# Run on 1xH200 (Northeastern HPC) — development/iteration mode
# Training runs without wallclock cap, TTT eval follows automatically

set -e

# Setup (run once)
# pip install torch numpy sentencepiece zstandard
# python3 data/cached_challenge_fineweb.py --variant sp1024

cd "$(dirname "$0")/../../.."

SEED=${1:-42}

echo "=== Starting Parameter Golf: SOTA + LoRA TTT ==="
echo "=== Seed: $SEED | Device: 1xGPU ==="

MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-0} \
SEED=$SEED \
RUN_ID="sota_ttt_seed${SEED}" \
TTT_ENABLED=1 \
TTT_LORA_RANK=${TTT_LORA_RANK:-8} \
TTT_LORA_LR=${TTT_LORA_LR:-0.01} \
TTT_CHUNK_SIZE=${TTT_CHUNK_SIZE:-256} \
TTT_BATCH_SIZE=${TTT_BATCH_SIZE:-64} \
TTT_GRAD_STEPS=${TTT_GRAD_STEPS:-1} \
torchrun --standalone --nproc_per_node=1 \
    records/track_10min_16mb/2026-03-22_SOTA_LoRA_TTT/train_gpt.py
