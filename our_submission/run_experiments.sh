#!/bin/bash
# Parameter Golf experiment runner
# Run on RunPod with 8xH100
# Each experiment takes ~10 minutes

set -e

DATA_PATH="./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
BASE_CMD="torchrun --standalone --nproc_per_node=8 our_submission/train_gpt.py"

run_experiment() {
    local name="$1"
    shift
    echo "=========================================="
    echo "EXPERIMENT: $name"
    echo "ENV: $@"
    echo "=========================================="

    RUN_ID="$name" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    TTT_ENABLED=1 \
    "$@" $BASE_CMD 2>&1 | tee "logs/${name}.log"

    echo "DONE: $name"
    echo ""
}

mkdir -p logs

# =====================================================
# PHASE 1: Reproduce SOTA baseline (verify we match 1.1194)
# =====================================================
echo "=== PHASE 1: BASELINE ==="
run_experiment "baseline_reproduce" \
    env LEAKY_SLOPE=0.5

# =====================================================
# PHASE 2: FP16 embedding passthrough (our key change)
# This should improve over baseline by avoiding int8 damage to tok_emb
# =====================================================
echo "=== PHASE 2: FP16 EMBEDDING ==="
run_experiment "fp16_embed" \
    env LEAKY_SLOPE=0.5

# =====================================================
# PHASE 3: Negative slope sweep
# SOTA uses 0.5. Test 0.3, 0.4, 0.6, 0.7
# =====================================================
echo "=== PHASE 3: LEAKY SLOPE SWEEP ==="
for slope in 0.3 0.4 0.45 0.55 0.6 0.7; do
    run_experiment "slope_${slope}" \
        env LEAKY_SLOPE=$slope
done

# =====================================================
# PHASE 4: Warmdown timing sweep
# SOTA uses 3500. Test 2500, 3000, 4000
# =====================================================
echo "=== PHASE 4: WARMDOWN SWEEP ==="
for wd in 2500 3000 4000; do
    run_experiment "warmdown_${wd}" \
        env LEAKY_SLOPE=0.5 WARMDOWN_ITERS=$wd
done

# =====================================================
# PHASE 5: Adaptive EMA decay
# SOTA uses fixed 0.997. Test ramp schedules
# =====================================================
echo "=== PHASE 5: EMA SWEEP ==="
run_experiment "ema_995_999" \
    env LEAKY_SLOPE=0.5 EMA_DECAY_START=0.995 EMA_DECAY_END=0.999

run_experiment "ema_993_999" \
    env LEAKY_SLOPE=0.5 EMA_DECAY_START=0.993 EMA_DECAY_END=0.999

run_experiment "ema_fixed_998" \
    env LEAKY_SLOPE=0.5 EMA_DECAY_START=0.998 EMA_DECAY_END=0.998

# =====================================================
# PHASE 6: Best combination
# Take the best slope + best warmdown + best EMA + FP16 embed
# =====================================================
echo "=== PHASE 6: BEST COMBO ==="
echo "Run this manually after reviewing Phase 1-5 results"
echo "Example:"
echo '  LEAKY_SLOPE=<best> WARMDOWN_ITERS=<best> EMA_DECAY_START=<best> EMA_DECAY_END=<best> ...'

echo ""
echo "ALL EXPERIMENTS COMPLETE"
echo "Review logs/ directory for results"
echo "Look for 'final_int6_sliding_window_exact' lines for final BPB scores"
