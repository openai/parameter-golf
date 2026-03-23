#!/bin/bash
# Run a specific experiment with 3 seeds for full evaluation
# Usage: ./run_full_3seed.sh <experiment_dir>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT="${1:?Usage: $0 <experiment_dir>}"

SEEDS=(42 1337 2024)

echo "=== Full 3-seed evaluation: $EXPERIMENT ==="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed $seed ---"
    "$SCRIPT_DIR/run_experiment.sh" "$EXPERIMENT" "$seed" 600
done

echo ""
echo "=== 3-Seed Summary for $EXPERIMENT ==="
for seed in "${SEEDS[@]}"; do
    LOG="$SCRIPT_DIR/$EXPERIMENT/train_seed${seed}.log"
    if [ -f "$LOG" ]; then
        BPB=$(grep "final_int8_zlib_roundtrip_exact" "$LOG" | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
        BYTES=$(grep "Serialized model int6" "$LOG" | grep -oP '[0-9]+ bytes' | head -1 || echo "N/A")
        echo "  Seed $seed: val_bpb=$BPB artifact=$BYTES"
    else
        echo "  Seed $seed: FAILED (no log)"
    fi
done
