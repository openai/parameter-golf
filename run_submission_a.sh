#!/bin/bash
# Run Option A submission on 8xH100 — 3 seeds for statistical significance
# NOTE: Requires flash_attn_interface (pip install flash-attn)
set -e
cd /workspace/parameter-golf

pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn may already be installed"

SCRIPT="records/track_10min_16mb/our_submission_a/train_gpt.py"
COMMON="NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64"

echo "=== Seed 42 ==="
eval "$COMMON SEED=42 RUN_ID=sub_a_seed42 torchrun --standalone --nproc_per_node=8 $SCRIPT" 2>&1 | tee logs/sub_a_seed42.log
echo ""

echo "=== Seed 1337 ==="
eval "$COMMON SEED=1337 RUN_ID=sub_a_seed1337 torchrun --standalone --nproc_per_node=8 $SCRIPT" 2>&1 | tee logs/sub_a_seed1337.log
echo ""

echo "=== Seed 2025 ==="
eval "$COMMON SEED=2025 RUN_ID=sub_a_seed2025 torchrun --standalone --nproc_per_node=8 $SCRIPT" 2>&1 | tee logs/sub_a_seed2025.log
echo ""

echo "=== RESULTS ==="
for seed in 42 1337 2025; do
    echo "Seed $seed:"
    grep "final_int8_zlib_roundtrip_exact" "logs/sub_a_seed${seed}.log" || echo "  (no result yet)"
done

echo ""
echo "Done! Copy logs and update submission.json with results."
