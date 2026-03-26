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

# --- G: 1+3×2+1, peri+birk, NO timestep — isolate gamma risk ---

run_experiment "Run G: 1+3x2+1 peri+birk (no timestep)" \
  env $COMMON $DATA RUN_ID=s2_G NUM_LAYERS=8 NUM_PRELUDE=1 NUM_SHARED=3 NUM_LOOPS=2 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=0 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

# --- H: 1+4×2+1, peri+birk, NO timestep — depth increase (10 eff layers) ---

run_experiment "Run H: 1+4x2+1 peri+birk (no timestep, 10 eff layers)" \
  env $COMMON $DATA RUN_ID=s2_H NUM_LAYERS=10 NUM_PRELUDE=1 NUM_SHARED=4 NUM_LOOPS=2 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=0 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

# --- I: 1+4×2+1, peri+birk+ts capped — does capped timestep help at depth? ---

run_experiment "Run I: 1+4x2+1 peri+birk+ts capped (GAMMA_MAX=4.0)" \
  env $COMMON $DATA RUN_ID=s2_I NUM_LAYERS=10 NUM_PRELUDE=1 NUM_SHARED=4 NUM_LOOPS=2 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 TIMESTEP_GAMMA_MAX=4.0 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

# --- J: 1+4×3+1, peri+birk, NO timestep — ambitious 14 eff layers, 3 loops ---

run_experiment "Run J: 1+4x3+1 peri+birk (no timestep, 14 eff layers, 3 loops)" \
  env $COMMON $DATA RUN_ID=s2_J NUM_LAYERS=14 NUM_PRELUDE=1 NUM_SHARED=4 NUM_LOOPS=3 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=0 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

# --- K: 1+4×3+1, peri+birk+ts capped — 3 loops + timestep scaling (best combo) ---

run_experiment "Run K: 1+4x3+1 peri+birk+ts capped (14 eff layers, 3 loops)" \
  env $COMMON $DATA RUN_ID=s2_K NUM_LAYERS=14 NUM_PRELUDE=1 NUM_SHARED=4 NUM_LOOPS=3 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 TIMESTEP_GAMMA_MAX=4.0 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

echo ""
echo "==============================="
echo "  FULL-SCALE SUMMARY"
echo "==============================="
echo "$SUMMARY"
echo "$FAILS run(s) failed."
