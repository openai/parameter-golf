#!/bin/bash
# H100 Experiment Runner — 4×H100, ~2 hours budget
# Created: 2026-03-22
#
# 4 GPUs = ~2x slower per step than 8 GPUs competition setup.
# 10-min wallclock per run → ~6,900 steps (vs 13,780 on 8×H100).
# With eval overhead (~3-5 min per run with sliding window stride=64),
# budget ~15 min per run → 8 runs in 2 hours.
#
# Priority-ordered: run top experiments first, skip later ones if time runs out.

set -euo pipefail

NGPUS=4
RESULTS_DIR="h100_results"
mkdir -p "$RESULTS_DIR"

# Adjust these paths to wherever data lives on the H100 machine
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

run_experiment() {
    local name="$1"
    shift
    echo ""
    echo "============================================="
    echo "  STARTING: $name at $(date)"
    echo "============================================="
    echo ""

    # Export all env vars passed as arguments
    for arg in "$@"; do
        export "$arg"
    done

    export RUN_ID="$name"
    export VOCAB_SIZE=1024
    export MAX_WALLCLOCK_SECONDS=600
    export TRAIN_BATCH_TOKENS=524288
    export TRAIN_SEQ_LEN=1024
    export VAL_LOSS_EVERY=200
    export TRAIN_LOG_EVERY=50

    torchrun --standalone --nproc_per_node=$NGPUS train_gpt_exp.py 2>&1 | tee "${RESULTS_DIR}/${name}.log"

    echo ""
    echo "=== $name DONE at $(date) ==="
    grep -E "(final_int|submission size|val_bpb)" "${RESULTS_DIR}/${name}.log" | tail -5
    echo ""

    # Unset experiment-specific vars to avoid leaking between runs
    for arg in "$@"; do
        unset "${arg%%=*}"
    done
}

echo "=== H100 experiment batch starting at $(date) ==="
echo "=== GPUs: $NGPUS, Budget: ~2 hours ==="
echo ""

# ─────────────────────────────────────────────────────────
# RUN 1: BASELINE — Reproduce official score (MANDATORY)
# Expected: ~1.24 BPB (fewer steps than 8-GPU, so slightly worse than 1.2244)
# ─────────────────────────────────────────────────────────
run_experiment "run1_baseline" \
    MLP_MULT=2 \
    WARMUP_STEPS=20

# ─────────────────────────────────────────────────────────
# RUN 2: Our proven best — 9L MLP3x + WD + OrthoInit + slide64
# Expected: ~1.20-1.23 BPB
# ─────────────────────────────────────────────────────────
run_experiment "run2_best_9L" \
    MLP_MULT=3 \
    WEIGHT_DECAY=0.04 \
    ORTHO_INIT=1 \
    COMPRESSOR=zstd \
    EVAL_SLIDING_WINDOW=1 \
    EVAL_WINDOW_STRIDE=64 \
    WARMUP_STEPS=20

# ─────────────────────────────────────────────────────────
# RUN 3: 10 Layers — more depth, proven config
# ~14% slower per step, ~6,000 steps in 10 min on 4 GPUs
# Expected: ~1.19-1.22 BPB
# ─────────────────────────────────────────────────────────
run_experiment "run3_10L" \
    NUM_LAYERS=10 \
    MLP_MULT=3 \
    WEIGHT_DECAY=0.04 \
    ORTHO_INIT=1 \
    COMPRESSOR=zstd \
    EVAL_SLIDING_WINDOW=1 \
    EVAL_WINDOW_STRIDE=64 \
    WARMUP_STEPS=20

# ─────────────────────────────────────────────────────────
# RUN 4: 11 Layers — best pre-quant val locally, untested clean
# ~25% slower per step, ~5,500 steps in 10 min on 4 GPUs
# Expected: ~1.18-1.21 BPB
# ─────────────────────────────────────────────────────────
run_experiment "run4_11L" \
    NUM_LAYERS=11 \
    MLP_MULT=3 \
    WEIGHT_DECAY=0.04 \
    ORTHO_INIT=1 \
    COMPRESSOR=zstd \
    EVAL_SLIDING_WINDOW=1 \
    EVAL_WINDOW_STRIDE=64 \
    WARMUP_STEPS=20

# ─────────────────────────────────────────────────────────
# RUN 5: 12 Layers — push depth boundary
# Check artifact size! If >16MB, this tells us the Int8 ceiling.
# Expected: ~1.17-1.20 BPB if fits
# ─────────────────────────────────────────────────────────
run_experiment "run5_12L" \
    NUM_LAYERS=12 \
    MLP_MULT=3 \
    WEIGHT_DECAY=0.04 \
    ORTHO_INIT=1 \
    COMPRESSOR=zstd \
    EVAL_SLIDING_WINDOW=1 \
    EVAL_WINDOW_STRIDE=64 \
    WARMUP_STEPS=20

# ─────────────────────────────────────────────────────────
# RUN 6: Best depth from runs 3-5 + BigramHash
# Using smaller table (1024) to fit Int8 budget
# ─────────────────────────────────────────────────────────
run_experiment "run6_10L_bigram" \
    NUM_LAYERS=10 \
    MLP_MULT=3 \
    WEIGHT_DECAY=0.04 \
    ORTHO_INIT=1 \
    BIGRAM_HASH_SIZE=1024 \
    COMPRESSOR=zstd \
    EVAL_SLIDING_WINDOW=1 \
    EVAL_WINDOW_STRIDE=64 \
    WARMUP_STEPS=20

# ─────────────────────────────────────────────────────────
# RUN 7: 11L + QAT Int6 + zstd (NO SWA) — if 11L/12L too big at Int8
# Int6 gives ~25% smaller artifact, more room for depth
# ─────────────────────────────────────────────────────────
run_experiment "run7_11L_int6" \
    NUM_LAYERS=11 \
    MLP_MULT=3 \
    WEIGHT_DECAY=0.04 \
    ORTHO_INIT=1 \
    QAT=1 \
    QUANT_BITS=6 \
    COMPRESSOR=zstd \
    EVAL_SLIDING_WINDOW=1 \
    EVAL_WINDOW_STRIDE=64 \
    WARMUP_STEPS=20

# ─────────────────────────────────────────────────────────
# RUN 8: Warmdown tuning — default 1200 is for 20K steps
# At ~6,900 steps (4 GPU), try 800
# ─────────────────────────────────────────────────────────
run_experiment "run8_warmdown" \
    NUM_LAYERS=10 \
    MLP_MULT=3 \
    WEIGHT_DECAY=0.04 \
    ORTHO_INIT=1 \
    WARMDOWN_ITERS=800 \
    COMPRESSOR=zstd \
    EVAL_SLIDING_WINDOW=1 \
    EVAL_WINDOW_STRIDE=64 \
    WARMUP_STEPS=20

echo ""
echo "============================================="
echo "  ALL RUNS COMPLETE at $(date)"
echo "============================================="
echo ""
echo "Results in ${RESULTS_DIR}/"
echo ""
echo "Quick summary:"
for f in ${RESULTS_DIR}/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep "final_int.*_roundtrip_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:[\d.]+' | tail -1)
    size=$(grep "Total submission size" "$f" 2>/dev/null | grep -oP '\d+ bytes' | tail -1)
    echo "  $name: $bpb ($size)"
done
