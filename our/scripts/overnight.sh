#!/bin/bash
# Overnight pipeline: reruns → adaptive sweep → full training on winner
# Expected: ~6-8 hours total
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

RESULTS="results/experiments.csv"
LOG="results/pipeline_log.txt"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

get_best() {
    tail -n +2 "$RESULTS" | sort -t',' -k4 -n | head -1
}

log "========================================="
log "OVERNIGHT PIPELINE START"
log "========================================="

# -----------------------------------------------
# PHASE 1: Rerun top candidates (10 shards, 600 iters)
# -----------------------------------------------
log "=== PHASE 1: Candidate reruns ==="
./our/scripts/rerun_top_candidates.sh 2>&1 | tee -a "$LOG"

log ""
log "Candidate reruns complete. Current best:"
log "$(get_best)"

# -----------------------------------------------
# PHASE 2: Adaptive sweep with fair comparisons
# Explore architecture shapes around the winner
# -----------------------------------------------
log ""
log "=== PHASE 2: Adaptive sweep (600 iters, 10 shards) ==="
python3 our/scripts/adaptive_sweep.py --max-experiments 15 2>&1

log ""
log "Adaptive sweep complete. Current best:"
log "$(get_best)"

# -----------------------------------------------
# PHASE 3: Full training on the winner
# 5000 iterations, full val eval, 10 shards
# -----------------------------------------------
log ""
log "=== PHASE 3: Full training on winner ==="

BEST_LINE=$(get_best)
BEST_NAME=$(echo "$BEST_LINE" | cut -d',' -f1)
BEST_BPB=$(echo "$BEST_LINE" | cut -d',' -f4)

# Extract config from the best experiment's log
BEST_LOG="results/logs/${BEST_NAME}.txt"
if [ -f "$BEST_LOG" ]; then
    B_DIM=$(grep "dim:" "$BEST_LOG" | head -1 | sed 's/.*dim:\([0-9]*\).*/\1/')
    B_HEADS=$(grep "heads:" "$BEST_LOG" | head -1 | sed 's/.*heads:\([0-9]*\).*/\1/')
    B_UNIQUE=$(grep "unique_layers:" "$BEST_LOG" | head -1 | sed 's/.*unique_layers:\([0-9]*\).*/\1/')
    B_RECUR=$(grep "recurrences:" "$BEST_LOG" | head -1 | sed 's/.*recurrences:\([0-9]*\).*/\1/')
fi
B_DIM=${B_DIM:-768}; B_HEADS=${B_HEADS:-12}; B_UNIQUE=${B_UNIQUE:-3}; B_RECUR=${B_RECUR:-3}

FULL_NAME="full_${B_UNIQUE}x${B_RECUR}_d${B_DIM}_h${B_HEADS}"
log "Winner: $BEST_NAME (BPB=$BEST_BPB)"
log "Running full training: $FULL_NAME (5000 iters, full val eval)"

SCRIPT=our/models/train_gpt_mlx_recurrent.py \
    ./our/scripts/run_experiment.sh "$FULL_NAME" \
    ITERATIONS=5000 \
    MAX_WALLCLOCK_SECONDS=0 \
    VAL_SAMPLE_FRAC=1.0 \
    NUM_UNIQUE_LAYERS=$B_UNIQUE \
    NUM_RECURRENCES=$B_RECUR \
    MODEL_DIM=$B_DIM \
    NUM_HEADS=$B_HEADS \
    NUM_KV_HEADS=4

log ""
log "========================================="
log "OVERNIGHT PIPELINE COMPLETE"
log "========================================="
log ""
log "Final leaderboard:"
tail -n +2 "$RESULTS" | sort -t',' -k4 -n | awk -F',' 'BEGIN{r=1}{printf "  %d. %-35s BPB=%-12s params=%s\n", r++, $1, $4, $3}' | tee -a "$LOG"
log ""
log "Full training result is your best BPB projection for the grant application."
