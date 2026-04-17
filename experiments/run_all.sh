#!/bin/bash
# Master runner: runs all experiments back-to-back.
# Deploy pod → SSH in → bash experiments/run_all.sh → come back to results.
# Auto-terminates pod when done.

set -uo pipefail

echo "========================================"
echo "=== Parameter Golf Experiment Sweep ==="
echo "=== $(date) ==="
echo "========================================"

# Setup
echo ""
echo "=== Phase 1: Setup ==="
bash experiments/setup_pod.sh

# Run planned experiments
EXPERIMENTS=(5 6 7 8 9 10 11 12 13 14)
RESULTS_FILE=/workspace/logs/sweep_results.txt
echo "Experiment | val_bpb (final) | Steps | Status" > "$RESULTS_FILE"

for n in "${EXPERIMENTS[@]}"; do
    script=$(ls experiments/run_exp${n}_*.sh 2>/dev/null | head -1)
    if [ -z "$script" ]; then
        echo "Skipping exp $n: no script found"
        continue
    fi

    echo ""
    echo "========================================"
    echo "=== Running Exp $n: $script ==="
    echo "=== $(date) ==="
    echo "========================================"

    if bash "$script"; then
        # Extract final val_bpb from log
        logfile=$(ls /workspace/logs/exp${n}_*.log 2>/dev/null | head -1)
        if [ -n "$logfile" ]; then
            final_bpb=$(grep "final_int8_zlib_roundtrip_exact" "$logfile" | grep -oP 'val_bpb:\K[0-9.]+' | tail -1)
            steps=$(grep "^step:" "$logfile" | tail -1 | grep -oP 'step:\K[0-9]+')
            echo "Exp $n | ${final_bpb:-MISSING} | ${steps:-?} | OK" >> "$RESULTS_FILE"
            echo ">>> Exp $n result: val_bpb=${final_bpb:-MISSING} steps=${steps:-?}"
        fi
    else
        echo "Exp $n | FAILED | - | ERROR" >> "$RESULTS_FILE"
        echo ">>> Exp $n FAILED, continuing..."
    fi
done

echo ""
echo "========================================"
echo "=== All planned experiments done ==="
echo "=== $(date) ==="
echo "========================================"
echo ""
echo "=== Summary ==="
cat "$RESULTS_FILE"

# Auto-terminate pod
echo ""
echo "=== Terminating pod ==="
if command -v runpodctl &>/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "Stopping pod $RUNPOD_POD_ID..."
    runpodctl stop pod "$RUNPOD_POD_ID"
else
    echo "WARNING: Cannot auto-terminate. Please terminate pod manually!"
fi
