#!/bin/bash
# Large vocab experiments.
# Tests SP16384 with aggressive embedding quantization.
set -euo pipefail

NPROC="${NPROC:-8}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"

export DATA_DIR="./data/"
export SLIDING_WINDOW_ENABLED=1
export MIXED_PRECISION_QUANT=1
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export COMPRESSOR=brotli

RESULTS_DIR="${SCRIPT_DIR}/vocab_results"
mkdir -p "$RESULTS_DIR"

run_experiment() {
    local name="$1"
    shift
    echo "=== Running: $name ==="
    env "$@" RUN_ID="$name" SEED=42 \
        torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
        2>&1 | tee "${RESULTS_DIR}/${name}.log"
}

# Baseline: SP8192 with int8 embeddings
run_experiment "sp8192_emb_int8" \
    VOCAB_SIZE=8192 EMBED_BITS=8 EMBED_CLIP_SIGMAS=20.0

# SP8192 with int6 embeddings (save space for bigger model)
run_experiment "sp8192_emb_int6" \
    VOCAB_SIZE=8192 EMBED_BITS=6 EMBED_CLIP_SIGMAS=12.85

# SP16384 with int6 embeddings
run_experiment "sp16384_emb_int6" \
    VOCAB_SIZE=16384 EMBED_BITS=6 EMBED_CLIP_SIGMAS=12.85

# SP16384 with int5 embeddings (aggressive)
run_experiment "sp16384_emb_int5" \
    VOCAB_SIZE=16384 EMBED_BITS=5 EMBED_CLIP_SIGMAS=10.0

echo "=== Vocab experiments complete ==="
