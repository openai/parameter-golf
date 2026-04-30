#!/usr/bin/env bash
# =============================================================================
# H100 EXPERIMENT SUITE — OpenAI Parameter Golf
# Runs 3 experiments sequentially on 8×H100, ~35 min total.
# Experiments:
#   A) MAIN     — 10L×960D, SwiGLU×3, CGGR decay 0.5→0.25, SWA, QAT  [submission candidate]
#   B) NO-CGGR  — identical to A but CGGR disabled                     [CGGR ablation]
#   C) NO-FFN   — 10L×1280D no MLP, CGGR decay, SWA, QAT              [FFN ablation, ~15.5MB]
#
# Usage:
#   bash run_h100_suite.sh               # run all 3
#   bash run_h100_suite.sh A B           # run specific experiments
# =============================================================================
set -uo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"
mkdir -p logs artifacts

NPROC="${NPROC_PER_NODE:-8}"
if [ $# -eq 0 ]; then EXPERIMENTS=(A B C); else EXPERIMENTS=("$@"); fi

# ---------------------------------------------------------------------------
# Data setup — download if not present
# ---------------------------------------------------------------------------
DATA_PATH="./data/datasets/fineweb10B_sp1024"
if [ ! -d "$DATA_PATH" ] || [ -z "$(ls -A "$DATA_PATH" 2>/dev/null)" ]; then
    echo "=== Downloading FineWeb dataset ==="
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

# ---------------------------------------------------------------------------
# Common hyperparameters shared across all runs
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    NUM_LAYERS=10
    ATTN_NHEADS=8
    ATTN_KV_HEADS=4
    ATTN_LINEAR_IMPL=casted
    MLP_ACTIVATION=swiglu
    BIGRAM_HASH_SIZE=10240
    BIGRAM_HASH_DIM=128
    SMEAR_GATE=1
    RESET_ON_BOS=1
    VOCAB_SIZE=1024
    TRAIN_SEQ_LEN=2048
    TRAIN_BATCH_TOKENS=786432
    ITERATIONS=20000
    WARMUP_STEPS=40
    WARMDOWN_ITERS=2000
    MAX_WALLCLOCK_SECONDS=590
    MATRIX_LR=0.02
    SCALAR_LR=0.04
    TIED_EMBED_LR=0.05
    EMBED_LR=0.6
    MUON_WD=0.04
    MUON_MOMENTUM=0.99
    MUON_MOMENTUM_WARMUP_START=0.92
    MUON_MOMENTUM_WARMUP_STEPS=1500
    CGGR_WARMUP=500
    SWA_START_FRACTION=0.7
    QAT_START_FRACTION=0.3
    EVAL_STRIDE=64
    COMPILE_MODEL=1
    VAL_LOSS_EVERY=500
    TRAIN_LOG_EVERY=100
    SEED=1337
)

# ---------------------------------------------------------------------------
# run_experiment <RUN_ID> <extra_env_key=val ...>
# ---------------------------------------------------------------------------
run_experiment() {
    local run_id="$1"; shift
    local log="$REPO/logs/${run_id}.txt"
    echo ""
    echo "============================================================"
    echo "  STARTING: $run_id"
    echo "  Log: $log"
    echo "  $(date)"
    echo "============================================================"

    local env_args=()
    for kv in "${COMMON_ARGS[@]}"; do env_args+=("$kv"); done
    for kv in "$@"; do env_args+=("$kv"); done

    env RUN_ID="$run_id" "${env_args[@]}" \
        torchrun --standalone --nproc_per_node="$NPROC" \
        -m train_gpt \
        2>&1 | tee "$log"

    echo ""
    echo "=== FINISHED: $run_id ==="
    echo "  sliding_window_bpb: $(grep '^final_sliding_window_exact' "$log" | grep -oP 'val_bpb:\K[0-9.]+' || echo 'N/A')"
    echo "  artifact_bytes:     $(grep '^Total submission size mixed' "$log" | grep -oP '\d+(?= bytes)' | head -1 || echo 'N/A')"
}

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
run_A() {
    # MAIN: 960D SwiGLU×3, CGGR decay 0.5→0.25, SWA 0.7, QAT 0.3
    # Confirmed local: 113M params, 14.22MB artifact, val_bpb=2.15 @ 300 steps
    run_experiment "h100_A_960d_cggr_decay" \
        MODEL_DIM=960 \
        ATTN_FFN_EXPAND=3.0 \
        CGGR_RATIO=0.5 \
        CGGR_RATIO_FINAL=0.25 \
        CGGR_DECAY_END=10000
}

run_B() {
    # NO-CGGR: identical to A but CGGR disabled (ratio=1.0 bypasses selection)
    # Isolates CGGR contribution to final bpb
    run_experiment "h100_B_960d_nocggr" \
        MODEL_DIM=960 \
        ATTN_FFN_EXPAND=3.0 \
        CGGR_RATIO=1.0
}

run_C() {
    # NO-FFN WIDER: 10L×1280D, no MLP (ATTN_FFN_EXPAND=0), CGGR decay
    # ~53M params, ~8MB artifact — room for larger model without FFN overhead
    # 1280D: ATTN_NHEADS=8 requires head_dim=160, divisible ✓
    # TRAIN_BATCH_TOKENS=786432 / (8 GPUs * 1 accum * 2048 seq_len) = 48 seqs/GPU ✓
    run_experiment "h100_C_1280d_noffn_cggr" \
        MODEL_DIM=1280 \
        ATTN_FFN_EXPAND=0.0 \
        CGGR_RATIO=0.5 \
        CGGR_RATIO_FINAL=0.25 \
        CGGR_DECAY_END=10000
}

# ---------------------------------------------------------------------------
# Execute requested experiments
# ---------------------------------------------------------------------------
for exp in "${EXPERIMENTS[@]}"; do
    case "$exp" in
        A) run_A ;;
        B) run_B ;;
        C) run_C ;;
        *) echo "Unknown experiment: $exp (valid: A B C)"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  EXPERIMENT SUITE COMPLETE — $(date)"
echo "============================================================"
printf "%-35s  %-12s  %-14s\n" "RUN" "SLIDING BPB" "ARTIFACT"
for run_id in h100_A_960d_cggr_decay h100_B_960d_nocggr h100_C_1280d_noffn_cggr; do
    log="$REPO/logs/${run_id}.txt"
    if [ -f "$log" ]; then
        bpb=$(grep '^final_sliding_window_exact' "$log" | grep -oP 'val_bpb:\K[0-9.]+' || echo 'N/A')
        art=$(grep '^Total submission size mixed' "$log" | grep -oP '\d+(?= bytes)' | head -1 || echo 'N/A')
        printf "%-35s  %-12s  %-14s\n" "$run_id" "$bpb" "$art"
    fi
done
echo ""
echo "Current SOTA: 1.1428 bpb (thwu1, mixed int5/int6 + BigramHash + SWA)"
