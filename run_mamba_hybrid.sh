#!/bin/bash
# =============================================================================
# Mamba-Attention Hybrid — GPU Training & Evaluation Pipeline
# =============================================================================
# Usage:
#   ./run_mamba_hybrid.sh phase1    # Smoke test (2 min)
#   ./run_mamba_hybrid.sh phase2a   # Train hybrid (10 min)
#   ./run_mamba_hybrid.sh phase2b   # Reproduce SOTA baseline (10 min)
#   ./run_mamba_hybrid.sh phase3    # Ablation experiments
#   ./run_mamba_hybrid.sh phase4    # 3-seed final (30 min)
#   ./run_mamba_hybrid.sh phase5    # Create submission package
#
# Prerequisites:
#   - 8xH100 80GB SXM (or 1xH100 for smoke test)
#   - pip install mamba-ssm>=2.2.0 causal-conv1d>=1.4.0
#   - Dataset: python3 data/cached_challenge_fineweb.py --variant sp1024
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Detect GPU count
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
echo "=== Detected $NUM_GPUS GPU(s) ==="

# Common env vars
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024

# =============================================================================
# Phase 1: Smoke Test + Go/No-Go
# =============================================================================
phase1() {
    echo "=== Phase 1: Smoke Test ==="

    # 1A: Verify CUDA kernels
    echo "--- 1A: Checking CUDA kernels ---"
    python3 -c "
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from causal_conv1d import causal_conv1d_fn
print('CUDA kernels loaded successfully')
" || { echo "CUDA kernel check failed. Install: pip install mamba-ssm>=2.2.0 causal-conv1d>=1.4.0"; exit 1; }

    # 1B: Step time benchmark (120 seconds)
    echo "--- 1B: Step time benchmark (120s) ---"
    MAMBA_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,15,16,17 \
    NUM_LAYERS=18 \
    MAMBA_D_STATE=32 \
    MAMBA_D_CONV=4 \
    MAMBA_EXPAND=1.5 \
    SEED=42 \
    MAX_WALLCLOCK_SECONDS=120 \
    RUN_ID=smoke_step_time \
    torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py 2>&1 | tee smoke_step_time.log

    echo ""
    echo "=== Go/No-Go Decision ==="
    echo "Check smoke_step_time.log for 'step_avg' value."
    echo "  < 90ms  → STRONG GO"
    echo "  90-100ms → GO"
    echo "  100-115ms → CONDITIONAL"
    echo "  > 115ms → NO-GO for record"
    echo ""
    grep -E "step_avg|step [0-9]+" smoke_step_time.log | tail -5
}

# =============================================================================
# Phase 2A: Train Our Hybrid (seed 42, full 600s)
# =============================================================================
phase2a() {
    echo "=== Phase 2A: Train Mamba-Attention Hybrid (seed 42) ==="
    MAMBA_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,15,16,17 \
    NUM_LAYERS=18 \
    MAMBA_D_STATE=32 \
    MAMBA_D_CONV=4 \
    MAMBA_EXPAND=1.5 \
    MAMBA_MATRIX_LR=0.015 \
    BIGRAM_VOCAB_SIZE=2048 \
    BIGRAM_DIM=128 \
    WARMDOWN_ITERS=3500 \
    TARGET_MB=15.9 \
    SEED=42 \
    RUN_ID=hybrid_18L_seed42 \
    torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py 2>&1 | tee train_hybrid_seed42.log

    echo ""
    echo "=== Phase 2A Results ==="
    grep -E "final_int6_sliding_window_exact|Total submission size|step_avg" train_hybrid_seed42.log | tail -5
}

# =============================================================================
# Phase 2B: Reproduce SOTA Baseline (seed 42)
# =============================================================================
phase2b() {
    echo "=== Phase 2B: Reproduce SOTA Baseline (seed 42) ==="
    cd records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072
    BIGRAM_VOCAB_SIZE=3072 \
    BIGRAM_DIM=112 \
    WARMDOWN_ITERS=4000 \
    TARGET_MB=15.9 \
    SEED=42 \
    RUN_ID=sota_baseline_seed42 \
    torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py 2>&1 | tee ../../../train_sota_seed42.log
    cd "$SCRIPT_DIR"

    echo ""
    echo "=== Phase 2B Results ==="
    grep -E "final_int6_sliding_window_exact|Total submission size|step_avg" train_sota_seed42.log | tail -5

    echo ""
    echo "=== Phase 2 Comparison ==="
    echo "Our hybrid:"
    grep "final_int6_sliding_window_exact" train_hybrid_seed42.log 2>/dev/null | tail -1
    echo "SOTA baseline:"
    grep "final_int6_sliding_window_exact" train_sota_seed42.log 2>/dev/null | tail -1
}

# =============================================================================
# Phase 3: Ablation Experiments
# =============================================================================
phase3() {
    echo "=== Phase 3: Ablation Experiments ==="
    mkdir -p ablation_logs

    # Helper function for ablation runs
    run_ablation() {
        local name="$1"; shift
        echo "--- Running: $name ---"
        RUN_ID="ablation_${name}" \
        SEED=42 \
        TARGET_MB=15.9 \
        BIGRAM_VOCAB_SIZE=2048 \
        BIGRAM_DIM=128 \
        "$@" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py 2>&1 | tee "ablation_logs/${name}.log"
    }

    # Tier 1: Architecture ablations

    # 3.1 All Mamba (0 attn)
    MAMBA_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
    NUM_LAYERS=18 MAMBA_D_STATE=32 MAMBA_EXPAND=1.5 WARMDOWN_ITERS=3500 \
    run_ablation "all_mamba_18L"

    # 3.1 2 Attn (top only)
    MAMBA_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    NUM_LAYERS=18 MAMBA_D_STATE=32 MAMBA_EXPAND=1.5 WARMDOWN_ITERS=3500 \
    run_ablation "2attn_top_18L"

    # 3.1 5 Attn (interleaved)
    MAMBA_LAYERS=0,1,2,4,5,6,8,9,10,12,13,15,16 \
    NUM_LAYERS=18 MAMBA_D_STATE=32 MAMBA_EXPAND=1.5 WARMDOWN_ITERS=3500 \
    run_ablation "5attn_interleaved_18L"

    # 3.2 Layer count: 12 layers
    MAMBA_LAYERS=0,1,2,3,4,5,6,7,8 \
    NUM_LAYERS=12 MAMBA_D_STATE=32 MAMBA_EXPAND=1.5 WARMDOWN_ITERS=3500 \
    run_ablation "12L_9M_3A"

    # 3.2 Layer count: 15 layers
    MAMBA_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11 \
    NUM_LAYERS=15 MAMBA_D_STATE=32 MAMBA_EXPAND=1.5 WARMDOWN_ITERS=3500 \
    run_ablation "15L_12M_3A"

    # 3.3 Attn position: bottom
    MAMBA_LAYERS=3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
    NUM_LAYERS=18 MAMBA_D_STATE=32 MAMBA_EXPAND=1.5 WARMDOWN_ITERS=3500 \
    run_ablation "3attn_bottom_18L"

    # 3.3 Attn position: top
    MAMBA_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
    NUM_LAYERS=18 MAMBA_D_STATE=32 MAMBA_EXPAND=1.5 WARMDOWN_ITERS=3500 \
    run_ablation "3attn_top_18L"

    # 3.3 Attn position: interleaved
    MAMBA_LAYERS=0,1,2,3,4,6,7,8,9,10,12,13,14,15,16 \
    NUM_LAYERS=18 MAMBA_D_STATE=32 MAMBA_EXPAND=1.5 WARMDOWN_ITERS=3500 \
    run_ablation "3attn_interleaved_18L"

    echo ""
    echo "=== Phase 3 Tier 1 Results ==="
    echo "Config | BPB | Artifact Bytes | Step Avg"
    echo "-------|-----|----------------|--------"
    for log in ablation_logs/*.log; do
        name=$(basename "$log" .log)
        bpb=$(grep "final_int6_sliding_window_exact val_bpb" "$log" 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
        art=$(grep "Total submission size" "$log" 2>/dev/null | tail -1 | grep -oP '\d+ bytes' | head -1)
        step=$(grep "step_avg" "$log" 2>/dev/null | tail -1 | grep -oP 'step_avg:\K[0-9.]+ms')
        echo "$name | ${bpb:-N/A} | ${art:-N/A} | ${step:-N/A}"
    done
}

# =============================================================================
# Phase 4: Three-Seed Final Evaluation
# =============================================================================
phase4() {
    echo "=== Phase 4: Three-Seed Final Evaluation ==="
    echo "Using locked config from Phase 3 ablation winner."
    echo ""

    # IMPORTANT: Update these env vars with the best config from Phase 3!
    # Default: our initial config (15M+3A, d_state=32, expand=1.5)
    BEST_MAMBA_LAYERS="${BEST_MAMBA_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,15,16,17}"
    BEST_NUM_LAYERS="${BEST_NUM_LAYERS:-18}"
    BEST_D_STATE="${BEST_D_STATE:-32}"
    BEST_EXPAND="${BEST_EXPAND:-1.5}"
    BEST_WARMDOWN="${BEST_WARMDOWN:-3500}"
    BEST_MAMBA_LR="${BEST_MAMBA_LR:-0.015}"

    for seed in 42 1337 2025; do
        echo "--- Seed $seed ---"
        MAMBA_LAYERS=$BEST_MAMBA_LAYERS \
        NUM_LAYERS=$BEST_NUM_LAYERS \
        MAMBA_D_STATE=$BEST_D_STATE \
        MAMBA_EXPAND=$BEST_EXPAND \
        MAMBA_MATRIX_LR=$BEST_MAMBA_LR \
        WARMDOWN_ITERS=$BEST_WARMDOWN \
        BIGRAM_VOCAB_SIZE=2048 \
        BIGRAM_DIM=128 \
        TARGET_MB=15.9 \
        SEED=$seed \
        RUN_ID="final_seed${seed}" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py 2>&1 | tee "train_seed${seed}.log"
    done

    echo ""
    echo "=== Phase 4: Statistical Test ==="
    python3 -c "
from scipy import stats
import re, sys

sota_nats = [1.88276292, 1.88156874, 1.88220393]
our_nats = []
for seed in [42, 1337, 2025]:
    with open(f'train_seed{seed}.log') as f:
        content = f.read()
    m = re.search(r'final_int6_sliding_window_exact val_loss:([0-9.]+)', content)
    if m:
        our_nats.append(float(m.group(1)))
    else:
        print(f'WARNING: Could not find val_loss for seed {seed}')

if len(our_nats) == 3:
    t_stat, p_val = stats.ttest_ind(our_nats, sota_nats, equal_var=False)
    mean_nats = sum(our_nats) / 3
    mean_bpb = mean_nats / 0.69314718  # ln(2)
    delta = mean_nats - sum(sota_nats) / 3
    print(f'Our nats (per seed): {our_nats}')
    print(f'Our mean nats: {mean_nats:.8f}')
    print(f'Our mean BPB:  {mean_bpb:.8f}')
    print(f'SOTA mean nats: {sum(sota_nats)/3:.8f}')
    print(f'Delta nats: {delta:.8f}')
    print(f't-statistic: {t_stat:.4f}')
    print(f'p-value: {p_val:.6f}')
    print()
    if delta <= -0.005 and p_val < 0.01:
        print('*** RECORD CRITERIA MET ***')
    elif delta <= 0:
        print('Better than SOTA but not statistically significant for record')
    else:
        print('Did not beat SOTA — submit as non-record')
else:
    print(f'Only found {len(our_nats)}/3 seed results')
"
}

# =============================================================================
# Phase 5: Submission Package
# =============================================================================
phase5() {
    echo "=== Phase 5: Create Submission Package ==="
    DATE=$(date +%Y-%m-%d)
    SUBMISSION_DIR="records/track_10min_16mb/${DATE}_MambaHybrid_18L_15M3A_Int6_GPTQ"

    mkdir -p "$SUBMISSION_DIR"
    cp train_gpt.py "$SUBMISSION_DIR/"
    cp requirements.txt "$SUBMISSION_DIR/"
    cp train_seed42.log "$SUBMISSION_DIR/" 2>/dev/null || echo "WARNING: train_seed42.log not found"
    cp train_seed1337.log "$SUBMISSION_DIR/" 2>/dev/null || echo "WARNING: train_seed1337.log not found"
    cp train_seed2025.log "$SUBMISSION_DIR/" 2>/dev/null || echo "WARNING: train_seed2025.log not found"

    echo "Submission folder created at: $SUBMISSION_DIR"
    echo "TODO: Create submission.json and README.md (see plan Phase 5B/5C)"
    ls -la "$SUBMISSION_DIR/"
}

# =============================================================================
# Main dispatcher
# =============================================================================
case "${1:-help}" in
    phase1)  phase1 ;;
    phase2a) phase2a ;;
    phase2b) phase2b ;;
    phase3)  phase3 ;;
    phase4)  phase4 ;;
    phase5)  phase5 ;;
    all)
        phase1
        echo ""; echo "=== Proceeding to Phase 2 ==="; echo ""
        phase2a
        phase2b
        echo ""; echo "=== Check Phase 2 results before Phase 3 ==="; echo ""
        ;;
    help|*)
        echo "Usage: $0 {phase1|phase2a|phase2b|phase3|phase4|phase5|all}"
        echo ""
        echo "Phases:"
        echo "  phase1   Smoke test + Go/No-Go (2 min)"
        echo "  phase2a  Train our hybrid (10 min)"
        echo "  phase2b  Reproduce SOTA baseline (10 min)"
        echo "  phase3   Ablation experiments (2-5 hours)"
        echo "  phase4   3-seed final evaluation (30 min)"
        echo "  phase5   Create submission package"
        echo "  all      Run phase1 + phase2a + phase2b"
        echo ""
        echo "Override GPU count: NUM_GPUS=1 $0 phase1"
        echo "Override best config: BEST_MAMBA_LAYERS=... $0 phase4"
        ;;
esac
