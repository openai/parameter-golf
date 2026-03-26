#!/bin/bash
# Full autonomous pipeline for Parameter Golf
# Runs architecture sweep → picks winner → tests enhancements
# Designed to run unattended (e.g., overnight or during a drive)
#
# Usage: ./pipeline.sh
# Expected runtime: ~4-6 hours on MacBook

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

RESULTS="experiments.csv"
LOGFILE="pipeline_log.txt"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOGFILE"
}

get_best_bpb() {
    if [ ! -f "$RESULTS" ]; then
        echo "999"
        return
    fi
    tail -n +2 "$RESULTS" | awk -F',' '{print $4}' | sort -n | head -1
}

get_best_experiment() {
    if [ ! -f "$RESULTS" ]; then
        echo "none"
        return
    fi
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
        local best=$(get_best_bpb)
        log "Finished: $name | BPB=$bpb | Best so far=$best"
    else
        log "WARNING: $name did not produce a log file"
    fi
}

already_done() {
    local name="$1"
    if [ ! -f "$RESULTS" ]; then
        return 1
    fi
    grep -q "^${name}," "$RESULTS" 2>/dev/null
}

log "========================================="
log "PARAMETER GOLF PIPELINE START"
log "========================================="

# =============================================================================
# PHASE 1: Focused architecture sweep (8 experiments, ~2 hours)
# =============================================================================
log ""
log "=== PHASE 1: Architecture Sweep ==="

# Key configs to test: vary layers, recurrences, and width
ARCH_CONFIGS=(
    "recur_2x4_d768_h12  3 NUM_UNIQUE_LAYERS=2 NUM_RECURRENCES=4 MODEL_DIM=768 NUM_HEADS=12"
    "recur_2x5_d768_h12  3 NUM_UNIQUE_LAYERS=2 NUM_RECURRENCES=5 MODEL_DIM=768 NUM_HEADS=12"
    "recur_3x4_d768_h12  3 NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=768 NUM_HEADS=12"
    "recur_3x3_d896_h8   3 NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3 MODEL_DIM=896 NUM_HEADS=8"
    "recur_3x4_d896_h8   3 NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=896 NUM_HEADS=8"
    "recur_4x3_d768_h12  3 NUM_UNIQUE_LAYERS=4 NUM_RECURRENCES=3 MODEL_DIM=768 NUM_HEADS=12"
    "recur_4x2_d896_h8   3 NUM_UNIQUE_LAYERS=4 NUM_RECURRENCES=2 MODEL_DIM=896 NUM_HEADS=8"
    "recur_2x5_d896_h8   3 NUM_UNIQUE_LAYERS=2 NUM_RECURRENCES=5 MODEL_DIM=896 NUM_HEADS=8"
)

for config in "${ARCH_CONFIGS[@]}"; do
    read -r name _dummy rest <<< "$config"
    if already_done "$name"; then
        log "Skipping $name (already completed)"
        continue
    fi
    # Parse the env vars
    env_args=$(echo "$config" | awk '{for(i=3;i<=NF;i++) printf "%s ", $i}')
    run_one "$name" "train_gpt_mlx_recurrent.py" ITERATIONS=200 $env_args
done

log ""
log "=== PHASE 1 COMPLETE ==="
log "Best architecture: $(get_best_experiment) at BPB=$(get_best_bpb)"

# =============================================================================
# PHASE 2: Test enhancements on best architecture (4 experiments, ~1 hour)
# =============================================================================
log ""
log "=== PHASE 2: Enhancement Tests ==="

# Extract best config params from the best experiment's log
BEST=$(get_best_experiment)
log "Building enhancements on top of: $BEST"

# Parse the best config's params from its log
if [ -f "logs/${BEST}.txt" ]; then
    BEST_DIM=$(grep "dim:" "logs/${BEST}.txt" | head -1 | sed 's/.*dim:\([0-9]*\).*/\1/')
    BEST_HEADS=$(grep "heads:" "logs/${BEST}.txt" | head -1 | sed 's/.*heads:\([0-9]*\).*/\1/')
    BEST_UNIQUE=$(grep "unique_layers:" "logs/${BEST}.txt" | head -1 | sed 's/.*unique_layers:\([0-9]*\).*/\1/')
    BEST_RECUR=$(grep "recurrences:" "logs/${BEST}.txt" | head -1 | sed 's/.*recurrences:\([0-9]*\).*/\1/')

    # Fallback to our known best if parsing fails
    BEST_DIM=${BEST_DIM:-768}
    BEST_HEADS=${BEST_HEADS:-12}
    BEST_UNIQUE=${BEST_UNIQUE:-3}
    BEST_RECUR=${BEST_RECUR:-3}
else
    BEST_DIM=768
    BEST_HEADS=12
    BEST_UNIQUE=3
    BEST_RECUR=3
fi

BASE_ARGS="NUM_UNIQUE_LAYERS=$BEST_UNIQUE NUM_RECURRENCES=$BEST_RECUR MODEL_DIM=$BEST_DIM NUM_HEADS=$BEST_HEADS NUM_KV_HEADS=4"

# Test 1: Enhanced (smear gate + backout + resid lambdas) - all tricks on
if ! already_done "enhanced_all"; then
    run_one "enhanced_all" "train_gpt_mlx_enhanced.py" ITERATIONS=200 $BASE_ARGS \
        USE_SMEAR_GATE=1 USE_BACKOUT=1
fi

# Test 2: Enhanced with only smear gate (isolate its contribution)
if ! already_done "enhanced_smear_only"; then
    run_one "enhanced_smear_only" "train_gpt_mlx_enhanced.py" ITERATIONS=200 $BASE_ARGS \
        USE_SMEAR_GATE=1 USE_BACKOUT=0
fi

# Test 3: Enhanced with only backout (isolate its contribution)
if ! already_done "enhanced_backout_only"; then
    run_one "enhanced_backout_only" "train_gpt_mlx_enhanced.py" ITERATIONS=200 $BASE_ARGS \
        USE_SMEAR_GATE=0 USE_BACKOUT=1
fi

# Test 4: Enhanced all tricks, longer training (500 iters)
if ! already_done "enhanced_all_500iter"; then
    run_one "enhanced_all_500iter" "train_gpt_mlx_enhanced.py" ITERATIONS=500 $BASE_ARGS \
        USE_SMEAR_GATE=1 USE_BACKOUT=1
fi

log ""
log "=== PHASE 2 COMPLETE ==="
log "Best overall: $(get_best_experiment) at BPB=$(get_best_bpb)"

# =============================================================================
# PHASE 3: Extended run on winner (1 experiment, ~1 hour)
# =============================================================================
log ""
log "=== PHASE 3: Extended Validation Run ==="

# Run the best enhanced config for 1000 iterations
if ! already_done "best_1000iter"; then
    FINAL_BEST=$(get_best_experiment)
    log "Running 1000-iter validation of approach from: $FINAL_BEST"

    # Use enhanced script with all tricks
    run_one "best_1000iter" "train_gpt_mlx_enhanced.py" ITERATIONS=1000 $BASE_ARGS \
        USE_SMEAR_GATE=1 USE_BACKOUT=1
fi

log ""
log "========================================="
log "PIPELINE COMPLETE"
log "========================================="
log ""
log "Final leaderboard:"
if [ -f "$RESULTS" ]; then
    echo "Experiment,BPB,Params" | tee -a "$LOGFILE"
    tail -n +2 "$RESULTS" | sort -t',' -k4 -n | awk -F',' '{printf "%-35s %s %s\n", $1, $4, $3}' | tee -a "$LOGFILE"
fi
log ""
log "Next steps:"
log "1. Review experiments.csv and pipeline_log.txt"
log "2. Best config ready for compute grant application"
log "3. Consider: larger vocab, QAT, test-time compute"
