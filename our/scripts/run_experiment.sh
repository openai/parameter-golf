#!/bin/bash
# Usage: ./our/scripts/run_experiment.sh <experiment_name> [KEY=VALUE ...]
#
# Examples:
#   ./our/scripts/run_experiment.sh baseline
#   SCRIPT=our/models/train_gpt_mlx_recurrent.py ./our/scripts/run_experiment.sh recurrent_test NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3 MODEL_DIM=768
#
# Defaults to 200 iterations, 1 shard, baseline MLX script.
# Set SCRIPT=our/models/train_gpt_mlx_recurrent.py for recurrent variant.

set -e

# Always work from repo root
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> [KEY=VALUE ...]"
    echo ""
    echo "Common configs:"
    echo "  SCRIPT=our/models/train_gpt_mlx_recurrent.py"
    echo "  ITERATIONS=500"
    echo "  NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3"
    echo "  MODEL_DIM=768 NUM_HEADS=12"
    exit 1
fi

EXPERIMENT_NAME="$1"
shift

# Defaults
export RUN_ID="$EXPERIMENT_NAME"
export ITERATIONS="${ITERATIONS:-600}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
export VAL_SAMPLE_FRAC="${VAL_SAMPLE_FRAC:-0.1}"  # 10% for fast A/B. Set 1.0 for full eval.
export OUT_DIR="results/logs"

# Script selection
SCRIPT="${SCRIPT:-train_gpt_mlx.py}"

# Apply user overrides
for arg in "$@"; do
    export "$arg"
done

echo "=== Experiment: $EXPERIMENT_NAME ==="
echo "Script: $SCRIPT"
echo "Iterations: $ITERATIONS"
echo "Model dim: ${MODEL_DIM:-512}"
echo "Val sample: ${VAL_SAMPLE_FRAC}"
echo ""

# Save full config snapshot
mkdir -p results/configs
cat > "results/configs/${EXPERIMENT_NAME}.json" << JSONEOF
{
    "experiment": "$EXPERIMENT_NAME",
    "script": "$SCRIPT",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "config": {
        "ITERATIONS": "$ITERATIONS",
        "TRAIN_BATCH_TOKENS": "$TRAIN_BATCH_TOKENS",
        "VAL_LOSS_EVERY": "$VAL_LOSS_EVERY",
        "VAL_BATCH_SIZE": "$VAL_BATCH_SIZE",
        "VAL_SAMPLE_FRAC": "$VAL_SAMPLE_FRAC",
        "MODEL_DIM": "${MODEL_DIM:-512}",
        "NUM_HEADS": "${NUM_HEADS:-8}",
        "NUM_KV_HEADS": "${NUM_KV_HEADS:-4}",
        "NUM_UNIQUE_LAYERS": "${NUM_UNIQUE_LAYERS:-}",
        "NUM_RECURRENCES": "${NUM_RECURRENCES:-}",
        "VOCAB_SIZE": "${VOCAB_SIZE:-1024}",
        "MLP_MULT": "${MLP_MULT:-2}",
        "USE_SMEAR_GATE": "${USE_SMEAR_GATE:-}",
        "USE_BACKOUT": "${USE_BACKOUT:-}",
        "MATRIX_LR": "${MATRIX_LR:-0.04}",
        "SCALAR_LR": "${SCALAR_LR:-0.04}",
        "TIED_EMBED_LR": "${TIED_EMBED_LR:-0.05}",
        "SEED": "${SEED:-1337}"
    }
}
JSONEOF

source .venv/bin/activate

# Run training
START_TIME=$(date +%s)
python3 "$SCRIPT"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Extract results
LOG_FILE="results/logs/${EXPERIMENT_NAME}.txt"
if [ -f "$LOG_FILE" ]; then
    VAL_BPB=$(grep "final_int8_zlib_roundtrip_exact" "$LOG_FILE" | tail -1 | sed 's/.*val_bpb:\([^ ]*\).*/\1/')
    VAL_LOSS=$(grep "final_int8_zlib_roundtrip_exact" "$LOG_FILE" | tail -1 | sed 's/.*val_loss:\([^ ]*\).*/\1/')
    PARAMS=$(grep "^model_params:" "$LOG_FILE" | head -1 | sed 's/model_params:\([^ ]*\).*/\1/')
    COMPRESSED=$(grep "^serialized_model_int8_zlib:" "$LOG_FILE" | head -1 | sed 's/serialized_model_int8_zlib:\([0-9]*\).*/\1/')
    UNIQUE_LAYERS=$(grep "unique_layers:" "$LOG_FILE" | head -1 | sed 's/.*unique_layers:\([0-9]*\).*/\1/')
    RECURRENCES=$(grep "recurrences:" "$LOG_FILE" | head -1 | sed 's/.*recurrences:\([0-9]*\).*/\1/')
    HEADS=$(grep " heads:" "$LOG_FILE" | head -1 | sed 's/.*heads:\([0-9]*\).*/\1/')

    UNIQUE_LAYERS="${UNIQUE_LAYERS:-0}"
    RECURRENCES="${RECURRENCES:-0}"
    HEADS="${HEADS:-${NUM_HEADS:-8}}"
    COMPRESSED="${COMPRESSED:-0}"

    # Append to results CSV
    RESULTS_FILE="results/experiments.csv"
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "experiment,script,params,val_bpb,val_loss,iterations,model_dim,num_heads,unique_layers,recurrences,compressed_bytes,elapsed_sec,date" > "$RESULTS_FILE"
    fi
    echo "${EXPERIMENT_NAME},${SCRIPT},${PARAMS},${VAL_BPB},${VAL_LOSS},${ITERATIONS},${MODEL_DIM:-512},${HEADS},${UNIQUE_LAYERS},${RECURRENCES},${COMPRESSED},${ELAPSED},$(date +%Y-%m-%d_%H:%M)" >> "$RESULTS_FILE"

    echo ""
    echo "=== Results ==="
    echo "val_bpb:      $VAL_BPB"
    echo "val_loss:     $VAL_LOSS"
    echo "params:       $PARAMS"
    echo "compressed:   $COMPRESSED bytes"
    echo "config:       unique=${UNIQUE_LAYERS} recur=${RECURRENCES} dim=${MODEL_DIM:-512} heads=${HEADS}"
    echo "elapsed:      ${ELAPSED}s"
    echo ""
    echo "Config:   results/configs/${EXPERIMENT_NAME}.json"
    echo "Log:      $LOG_FILE"
else
    echo "WARNING: Log file not found at $LOG_FILE"
fi
