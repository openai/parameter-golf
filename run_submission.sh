#!/bin/bash
# Run submission on 8xH100 — 3 seeds for statistical significance
set -e
cd /workspace/parameter-golf

SCRIPT="records/track_10min_16mb/2026-03-27_LeakyReLU2_EMA_QAT_BigramHash20K_MLP35x/train_gpt.py"
COMMON="NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64"

echo "=== Seed 42 ==="
eval "$COMMON SEED=42 RUN_ID=sub_b_seed42 torchrun --standalone --nproc_per_node=8 $SCRIPT" 2>&1 | tee logs/sub_b_seed42.log
echo ""

echo "=== Seed 1337 ==="
eval "$COMMON SEED=1337 RUN_ID=sub_b_seed1337 torchrun --standalone --nproc_per_node=8 $SCRIPT" 2>&1 | tee logs/sub_b_seed1337.log
echo ""

echo "=== Seed 2025 ==="
eval "$COMMON SEED=2025 RUN_ID=sub_b_seed2025 torchrun --standalone --nproc_per_node=8 $SCRIPT" 2>&1 | tee logs/sub_b_seed2025.log
echo ""

echo "=== RESULTS ==="
for seed in 42 1337 2025; do
    echo "Seed $seed:"
    grep "final_int8_zlib_roundtrip_exact" "logs/sub_b_seed${seed}.log" || echo "  (no result yet)"
done

echo ""
echo "Done! Copy logs and update submission.json with results."
