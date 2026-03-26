#!/bin/bash
# Phase 1B: Budget-maximizing architecture sweep
# These configs fill 80-95% of the 16MB budget
# Run AFTER pipeline.sh finishes (or independently)
#
# Usage: ./pipeline_maxbudget.sh

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

RESULTS="experiments.csv"
LOGFILE="pipeline_log.txt"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOGFILE"
}

get_best_bpb() {
    if [ ! -f "$RESULTS" ]; then echo "999"; return; fi
    tail -n +2 "$RESULTS" | awk -F',' '{print $4}' | sort -n | head -1
}

get_best_experiment() {
    if [ ! -f "$RESULTS" ]; then echo "none"; return; fi
    tail -n +2 "$RESULTS" | sort -t',' -k4 -n | head -1 | cut -d',' -f1
}

run_one() {
    local name="$1"
    local script="$2"
    shift 2
    log "Starting: $name (script=$script)"
    SCRIPT="$script" ./run_experiment.sh "$name" "$@" 2>&1 | tail -5
    if [ -f "logs/${name}.txt" ]; then
        local bpb=$(grep "final_int8_zlib_roundtrip_exact" "logs/${name}.txt" | tail -1 | sed 's/.*val_bpb:\([^ ]*\).*/\1/')
        log "Finished: $name | BPB=$bpb | Best so far=$(get_best_bpb)"
    else
        log "WARNING: $name did not produce a log file"
    fi
}

already_done() {
    [ -f "$RESULTS" ] && grep -q "^${1}," "$RESULTS" 2>/dev/null
}

log ""
log "========================================="
log "BUDGET-MAXIMIZING SWEEP START"
log "========================================="
log "Current best: $(get_best_experiment) at BPB=$(get_best_bpb)"

# Configs that use 80-95% of 16MB budget
# These are the ones most likely to win since they use the most capacity

CONFIGS=(
    # 95% budget — 3 unique layers, very wide (d1152)
    "max_3x3_d1152_h12  train_gpt_mlx_recurrent.py NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3 MODEL_DIM=1152 NUM_HEADS=12"
    "max_3x4_d1152_h12  train_gpt_mlx_recurrent.py NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=1152 NUM_HEADS=12"
    # 93% budget — 5 unique layers, wide (d896)
    "max_5x2_d896_h16   train_gpt_mlx_recurrent.py NUM_UNIQUE_LAYERS=5 NUM_RECURRENCES=2 MODEL_DIM=896 NUM_HEADS=16"
    "max_5x3_d896_h16   train_gpt_mlx_recurrent.py NUM_UNIQUE_LAYERS=5 NUM_RECURRENCES=3 MODEL_DIM=896 NUM_HEADS=16"
    # 88% budget — 6 unique layers (most diversity)
    "max_6x2_d768_h12   train_gpt_mlx_recurrent.py NUM_UNIQUE_LAYERS=6 NUM_RECURRENCES=2 MODEL_DIM=768 NUM_HEADS=12"
    "max_6x3_d768_h12   train_gpt_mlx_recurrent.py NUM_UNIQUE_LAYERS=6 NUM_RECURRENCES=3 MODEL_DIM=768 NUM_HEADS=12"
    # 80% budget — 3 unique layers, d1024
    "max_3x3_d1024_h8   train_gpt_mlx_recurrent.py NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3 MODEL_DIM=1024 NUM_HEADS=8"
    "max_3x4_d1024_h8   train_gpt_mlx_recurrent.py NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=1024 NUM_HEADS=8"
)

for config in "${CONFIGS[@]}"; do
    read -r name script rest <<< "$config"
    if already_done "$name"; then
        log "Skipping $name (already completed)"
        continue
    fi
    run_one "$name" "$script" ITERATIONS=200 $rest
done

log ""
log "========================================="
log "BUDGET-MAXIMIZING SWEEP COMPLETE"
log "========================================="
log "Best overall: $(get_best_experiment) at BPB=$(get_best_bpb)"
log ""
log "Leaderboard:"
if [ -f "$RESULTS" ]; then
    tail -n +2 "$RESULTS" | sort -t',' -k4 -n | awk -F',' '{printf "  %-35s BPB=%-12s params=%s\n", $1, $4, $3}'
fi

# Now run enhancements on the overall best
BEST=$(get_best_experiment)
BEST_LOG="logs/${BEST}.txt"
if [ -f "$BEST_LOG" ]; then
    BEST_DIM=$(grep "dim:" "$BEST_LOG" | head -1 | sed 's/.*dim:\([0-9]*\).*/\1/')
    BEST_HEADS=$(grep "heads:" "$BEST_LOG" | head -1 | sed 's/.*heads:\([0-9]*\).*/\1/')
    BEST_UNIQUE=$(grep "unique_layers:" "$BEST_LOG" | head -1 | sed 's/.*unique_layers:\([0-9]*\).*/\1/')
    BEST_RECUR=$(grep "recurrences:" "$BEST_LOG" | head -1 | sed 's/.*recurrences:\([0-9]*\).*/\1/')
fi
BEST_DIM=${BEST_DIM:-768}; BEST_HEADS=${BEST_HEADS:-12}; BEST_UNIQUE=${BEST_UNIQUE:-3}; BEST_RECUR=${BEST_RECUR:-3}
BASE_ARGS="NUM_UNIQUE_LAYERS=$BEST_UNIQUE NUM_RECURRENCES=$BEST_RECUR MODEL_DIM=$BEST_DIM NUM_HEADS=$BEST_HEADS NUM_KV_HEADS=4"

log ""
log "=== ENHANCEMENT TESTS on $BEST ==="

if ! already_done "maxbudget_enhanced_all"; then
    run_one "maxbudget_enhanced_all" "train_gpt_mlx_enhanced.py" ITERATIONS=200 $BASE_ARGS USE_SMEAR_GATE=1 USE_BACKOUT=1
fi

if ! already_done "maxbudget_enhanced_500iter"; then
    run_one "maxbudget_enhanced_500iter" "train_gpt_mlx_enhanced.py" ITERATIONS=500 $BASE_ARGS USE_SMEAR_GATE=1 USE_BACKOUT=1
fi

log ""
log "========================================="
log "ALL DONE"
log "========================================="
log "Final best: $(get_best_experiment) at BPB=$(get_best_bpb)"
