#!/bin/bash
set -uo pipefail

SCRIPT="train_gpt.py"
NGPU=${NGPU:-1}
COMMON="SEED=1337 ITERATIONS=2000 VAL_LOSS_EVERY=500 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200"
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

# --- Baselines (leaky_relu² matched) --- run first for early signal ---

run_experiment "Run A': 8L standard (leaky_relu² baseline)" \
  env $COMMON $DATA RUN_ID=s1_Ap NUM_LAYERS=8 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

run_experiment "Run B': 4x2 bare recurrence (leaky_relu² baseline)" \
  env $COMMON $DATA RUN_ID=s1_Bp NUM_LAYERS=8 NUM_SHARED=4 NUM_LOOPS=2 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

# --- Experimental runs ---

run_experiment "Run C: 4x2 recurrence + peri-norm + birkhoff mix" \
  env $COMMON $DATA RUN_ID=s1_C NUM_LAYERS=8 NUM_SHARED=4 NUM_LOOPS=2 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

run_experiment "Run C': 4x2 recurrence + birkhoff only (no peri-norm)" \
  env $COMMON $DATA RUN_ID=s1_Cp NUM_LAYERS=8 NUM_SHARED=4 NUM_LOOPS=2 \
  USE_PERI_NORM=0 USE_BIRKHOFF_MIX=1 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

run_experiment "Run D: 4x2 recurrence + peri-norm + birkhoff + timestep scaling" \
  env $COMMON $DATA RUN_ID=s1_D NUM_LAYERS=8 NUM_SHARED=4 NUM_LOOPS=2 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

run_experiment "Run E: 1 prelude + 3x2 shared + 1 coda + all fixes" \
  env $COMMON $DATA RUN_ID=s1_E NUM_LAYERS=8 NUM_PRELUDE=1 NUM_SHARED=3 NUM_LOOPS=2 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

run_experiment "Run F: 1 prelude + 2x3 shared + 1 coda + all fixes (3 loops!)" \
  env $COMMON $DATA RUN_ID=s1_F NUM_LAYERS=8 NUM_PRELUDE=1 NUM_SHARED=2 NUM_LOOPS=3 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

echo ""
echo "==============================="
echo "  SCREENING SUMMARY"
echo "==============================="
echo "$SUMMARY"
echo "$FAILS run(s) failed."
