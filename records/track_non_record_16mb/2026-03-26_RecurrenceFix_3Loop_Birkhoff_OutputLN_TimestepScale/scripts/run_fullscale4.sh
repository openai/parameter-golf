#!/bin/bash
set -uo pipefail

SCRIPT="train_gpt.py"
NGPU=${NGPU:-8}
COMMON="SEED=1337 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=200"
DATA="DATA_PATH=${DATA_PATH:-./data/datasets/fineweb10B_sp1024} TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024"

FAILS=0
SUMMARY=""

run_experiment() {
    local name="$1"; shift
    echo ""
    echo "=== $name ==="
    if "$@"; then
        SUMMARY="${SUMMARY}  PASS  $name"$'\n'
    else
        SUMMARY="${SUMMARY}  FAIL  $name (exit $?)"$'\n'
        FAILS=$((FAILS + 1))
    fi
}

# --- T: 1+4×3+1, sinusoidal depth encoding + FiLM bias, 3 loops (compare vs s3_O and s4_R) ---

run_experiment "Run T: 1+4x3+1 sinusoidal depth enc + FiLM bias (3 loops)" \
  env $COMMON $DATA RUN_ID=s5_T NUM_LAYERS=14 NUM_PRELUDE=1 NUM_SHARED=4 NUM_LOOPS=3 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 TIMESTEP_GAMMA_MAX=4.0 \
  USE_TIMESTEP_BIAS=1 USE_DEPTH_EMBED=1 USE_UNIQUE_NORMS=0 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

echo ""
echo "==============================="
echo "  FULL-SCALE 4 SUMMARY"
echo "==============================="
echo "$SUMMARY"
echo "$FAILS run(s) failed."
