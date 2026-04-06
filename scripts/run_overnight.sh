#!/usr/bin/env bash
# Parameter Golf — Overnight Experiment Runner
# Runs ~20 experiments sequentially (~3.5 hours total)
#
# Usage:
#   cd ~/parameter-golf
#   nohup bash scripts/run_overnight.sh 2>&1 | tee experiments/logs/overnight_$(date +%Y%m%d).log &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1
TRAIN="python3 training/train.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="experiments/results/overnight_${TIMESTAMP}.csv"
TRAIN_SECONDS=${TRAIN_SECONDS:-600}

mkdir -p experiments/results experiments/logs experiments/checkpoints

echo "================================================================"
echo "  Parameter Golf — Overnight Experiments"
echo "  Started: $(date)"
echo "  Train time per experiment: ${TRAIN_SECONDS}s"
echo "  Results: ${RESULTS_FILE}"
echo "================================================================"
echo ""

# Summary collector
echo "exp_name,model_type,d_model,layers_or_sharing,val_bpb,elapsed,status" > "$RESULTS_FILE"

run_experiment() {
    local exp_name="$1"
    shift
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT: $exp_name"
    echo "  Started: $(date)"
    echo "================================================================"

    local start_time=$(date +%s)

    if $TRAIN --exp-name "$exp_name" --train-seconds "$TRAIN_SECONDS" "$@" 2>&1; then
        local status="OK"
    else
        local status="FAILED"
        echo "  [!] Experiment $exp_name FAILED, continuing..."
    fi

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Extract val_bpb from CSV if available
    local csv_file="experiments/results/${exp_name}.csv"
    local val_bpb="N/A"
    if [[ -f "$csv_file" ]]; then
        val_bpb=$(tail -1 "$csv_file" | awk -F',' '{print $6}' | tr -d ' ')
    fi

    echo "$exp_name,$*,$val_bpb,$elapsed,$status" >> "$RESULTS_FILE"
    echo "  -> $exp_name: val_bpb=$val_bpb, ${elapsed}s, $status"
}

# ================================================================
# EXPERIMENT 1: Baseline (no sharing)
# ================================================================
echo ""
echo "======== PHASE 1: BASELINE ========"
run_experiment "baseline_10layer_d512" \
    --model-type standard \
    --n-layers 10 \
    --d-model 512 \
    --n-heads 8

# ================================================================
# EXPERIMENT 2: Weight sharing sweep
# ================================================================
echo ""
echo "======== PHASE 2: WEIGHT SHARING SWEEP ========"

run_experiment "shared_3x7_d512" \
    --model-type shared \
    --n-unique-layers 3 --n-loops 7 \
    --d-model 512 --n-heads 8

run_experiment "shared_4x5_d512" \
    --model-type shared \
    --n-unique-layers 4 --n-loops 5 \
    --d-model 512 --n-heads 8

run_experiment "shared_5x4_d512" \
    --model-type shared \
    --n-unique-layers 5 --n-loops 4 \
    --d-model 512 --n-heads 8

run_experiment "shared_6x3_d512" \
    --model-type shared \
    --n-unique-layers 6 --n-loops 3 \
    --d-model 512 --n-heads 8

run_experiment "shared_7x3_d512" \
    --model-type shared \
    --n-unique-layers 7 --n-loops 3 \
    --d-model 512 --n-heads 8

run_experiment "shared_10x2_d512" \
    --model-type shared \
    --n-unique-layers 10 --n-loops 2 \
    --d-model 512 --n-heads 8

# ================================================================
# EXPERIMENT 3: Width sweep (with 5x4 sharing)
# ================================================================
echo ""
echo "======== PHASE 3: WIDTH SWEEP ========"

run_experiment "shared_5x4_d384" \
    --model-type shared \
    --n-unique-layers 5 --n-loops 4 \
    --d-model 384 --n-heads 8

run_experiment "shared_5x4_d448" \
    --model-type shared \
    --n-unique-layers 5 --n-loops 4 \
    --d-model 448 --n-heads 8

# d512 already done above

run_experiment "shared_5x4_d576" \
    --model-type shared \
    --n-unique-layers 5 --n-loops 4 \
    --d-model 576 --n-heads 8

run_experiment "shared_5x4_d640" \
    --model-type shared \
    --n-unique-layers 5 --n-loops 4 \
    --d-model 640 --n-heads 8

# ================================================================
# EXPERIMENT 4: MLP ratio sweep (5x4, d=512)
# ================================================================
echo ""
echo "======== PHASE 4: MLP RATIO SWEEP ========"

run_experiment "shared_5x4_d512_mlp2.0" \
    --model-type shared \
    --n-unique-layers 5 --n-loops 4 \
    --d-model 512 --n-heads 8 --mlp-ratio 2.0

run_experiment "shared_5x4_d512_mlp2.5" \
    --model-type shared \
    --n-unique-layers 5 --n-loops 4 \
    --d-model 512 --n-heads 8 --mlp-ratio 2.5

# mlp3.0 already done above

run_experiment "shared_5x4_d512_mlp3.5" \
    --model-type shared \
    --n-unique-layers 5 --n-loops 4 \
    --d-model 512 --n-heads 8 --mlp-ratio 3.5

run_experiment "shared_5x4_d512_mlp4.0" \
    --model-type shared \
    --n-unique-layers 5 --n-loops 4 \
    --d-model 512 --n-heads 8 --mlp-ratio 4.0

# ================================================================
# EXPERIMENT 5: LR Finder on best config
# ================================================================
echo ""
echo "======== PHASE 5: LR FINDER ========"
echo "Running LR finder on 5x4 d512 config..."
python3 training/lr_finder.py \
    --model-type shared \
    --n-unique-layers 5 --n-loops 4 \
    --d-model 512 --n-heads 8 \
    --exp-name "lr_finder_5x4_d512" \
    2>&1 || echo "[!] LR finder failed, continuing..."

# ================================================================
# SUMMARY
# ================================================================
echo ""
echo "================================================================"
echo "  OVERNIGHT EXPERIMENTS COMPLETE"
echo "  Finished: $(date)"
echo "  Results: ${RESULTS_FILE}"
echo "================================================================"
echo ""

if [[ -f "$RESULTS_FILE" ]]; then
    echo "SUMMARY:"
    echo "--------"
    column -t -s',' "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"
fi
