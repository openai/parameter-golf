#!/bin/bash
# ttt_sweep.sh — Run 3 TTT configs in sequence against a trained checkpoint.
# Usage: MODEL_PATH=/path/to/final_model.pt bash experiments/Rascal_Stripper/ttt_sweep.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC=${NPROC:-8}
MODEL_PATH=${MODEL_PATH:-""}

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH is required."
    echo "Usage: MODEL_PATH=/path/to/final_model.pt bash $0"
    exit 1
fi

LOG_DIR="$SCRIPT_DIR/ttt_sweep_logs"
mkdir -p "$LOG_DIR"

CALIBRATE="$SCRIPT_DIR/ttt_calibrate.py"

run_ttt() {
    local name=$1
    local lr=$2
    local epochs=$3
    local freeze=$4
    local chunk=$5
    local log="$LOG_DIR/${name}.log"

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  TTT: $name"
    echo "  lr=$lr  epochs=$epochs  freeze_blocks=$freeze  chunk=$chunk"
    echo "══════════════════════════════════════════════════"

    MODEL_PATH="$MODEL_PATH" \
    TTT_LR="$lr" \
    TTT_EPOCHS="$epochs" \
    TTT_FREEZE_BLOCKS="$freeze" \
    TTT_CHUNK_TOKENS="$chunk" \
    torchrun --nproc_per_node="$NPROC" "$CALIBRATE" \
        2>&1 | tee "$log" | grep --line-buffered -E "baseline_sliding|ttt_result|ttt_delta|BETTER|WORSE|ttt_chunk \[|SUMMARY|baseline :|ttt     :|delta"

    echo "  ── $name done ──"
}

# Config A: conservative — low LR, 1 epoch, big chunks (least risk of overfitting)
run_ttt "A_conservative"  0.0001  1  2  65536

# Config B: balanced — medium LR, 2 epochs, standard chunks
run_ttt "B_balanced"      0.0001  2  2  32768

# Config C: aggressive — original LR, 3 epochs, standard chunks
run_ttt "C_aggressive"    0.0005  3  2  32768

echo ""
echo "══════════════════════════════════════════════════"
echo "  SWEEP RESULTS — TTT delta vs baseline"
echo "══════════════════════════════════════════════════"
for name in A_conservative B_balanced C_aggressive; do
    log="$LOG_DIR/${name}.log"
    baseline=$(grep "baseline_sliding" "$log" 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    ttt=$(grep "ttt_result" "$log" 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    delta=$(grep "ttt_delta" "$log" 2>/dev/null | tail -1 | grep -oP 'bpb:\K[+-][0-9.]+' || echo "N/A")
    verdict=$(grep -oE "BETTER|WORSE" "$log" 2>/dev/null | tail -1 || echo "N/A")
    printf "  %-16s  baseline=%-12s  ttt=%-12s  delta=%-12s  %s\n" "$name" "$baseline" "$ttt" "$delta" "$verdict"
done
echo "══════════════════════════════════════════════════"
