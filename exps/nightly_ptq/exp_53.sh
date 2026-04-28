#!/bin/bash
export RUN_ID="nightly_53_wd_1600__SEQ_LEN_512"
export EXPERIMENT_NAME="Nightly 53: wd_1600__SEQ_LEN_512"

# Base config from ptq_int5mlp_L20_d288.sh
export NUM_LAYERS=20
export MODEL_DIM=288
export MLP_MULT=4
export TERNARY_ENABLED=0
export QAT_BITS=0
export PTQ_BITS=6
export PTQ_MLP_BITS=5
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export OPTIMIZER=muon_adam
export LR_SCHEDULE=trapezoid

# Hypotheses overrides
export WARMDOWN_ITERS="1600"
export TRAIN_SEQ_LEN="512"

# Validation overrides (controlled by run_all_nightly.sh)
export ITERATIONS="${NIGHTLY_ITERATIONS:-20000}"
export MAX_WALLCLOCK_SECONDS="${NIGHTLY_WALLCLOCK:-600}"
if [ -n "$NIGHTLY_COMET_KEY" ]; then
    export COMET_API_KEY="$NIGHTLY_COMET_KEY"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

torchrun --nproc_per_node=8 train_gpt.py
