#!/bin/bash
# Final submission run on 8xH100 SXM — 10-min wallclock cap enforced
# Run 3 seeds for statistical significance

set -e

cd "$(dirname "$0")/../../.."

for SEED in 42 1337 2024; do
    echo "=== Seed: $SEED ==="
    SEED=$SEED \
    RUN_ID="sota_ttt_seed${SEED}" \
    TTT_ENABLED=1 \
    TTT_LORA_RANK=${TTT_LORA_RANK:-8} \
    TTT_LORA_LR=${TTT_LORA_LR:-0.01} \
    TTT_CHUNK_SIZE=${TTT_CHUNK_SIZE:-256} \
    TTT_BATCH_SIZE=${TTT_BATCH_SIZE:-64} \
    TTT_GRAD_STEPS=${TTT_GRAD_STEPS:-1} \
    torchrun --standalone --nproc_per_node=8 \
        records/track_10min_16mb/2026-03-22_SOTA_LoRA_TTT/train_gpt.py
    echo "=== Done seed $SEED ==="
    echo ""
done
