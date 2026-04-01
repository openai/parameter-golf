#!/bin/bash
# Rerun top candidates with 10 shards + 600 iterations
# Fair comparison across different param sizes with enough data
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

SCRIPT="our/models/train_gpt_mlx_recurrent.py"
COMMON="ITERATIONS=600 VAL_SAMPLE_FRAC=0.1"

echo "=== Rerunning top candidates with 10 shards, 600 iters ==="
echo ""

# 1. Current best: 3x3 d768 (12.6M params, 6.9MB)
echo "[1/4] 3x3_d768_h12"
SCRIPT=$SCRIPT ./our/scripts/run_experiment.sh rerun_3x3_d768_h12 \
    $COMMON NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3 MODEL_DIM=768 NUM_HEADS=12

# 2. Moderate: 3x3 d896 (17.8M params, 9.8MB)
echo "[2/4] 3x3_d896_h8"
SCRIPT=$SCRIPT ./our/scripts/run_experiment.sh rerun_3x3_d896_h8 \
    $COMMON NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3 MODEL_DIM=896 NUM_HEADS=8

# 3. Large: 3x3 d1024 (23M params, 11.9MB)
echo "[3/4] 3x3_d1024_h8"
SCRIPT=$SCRIPT ./our/scripts/run_experiment.sh rerun_3x3_d1024_h8 \
    $COMMON NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3 MODEL_DIM=1024 NUM_HEADS=8

# 4. Max budget: 3x3 d1152 (27.7M params, ~15.2MB)
echo "[4/4] 3x3_d1152_h12"
SCRIPT=$SCRIPT ./our/scripts/run_experiment.sh rerun_3x3_d1152_h12 \
    $COMMON NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3 MODEL_DIM=1152 NUM_HEADS=12

echo ""
echo "=== All done! ==="
echo ""
echo "Results:"
tail -n +1 results/experiments.csv | grep "rerun_" | sort -t',' -k4 -n | \
    awk -F',' '{printf "  %-25s BPB=%-12s params=%s compressed=%s\n", $1, $4, $3, $11}'
