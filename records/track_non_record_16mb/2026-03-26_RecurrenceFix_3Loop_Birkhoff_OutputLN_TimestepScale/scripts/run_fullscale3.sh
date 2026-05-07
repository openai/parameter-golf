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

# --- P: 1+4×2+1, full passthrough stack (depth embed + unique norms), 2 loops (compare vs s3_N) ---

run_experiment "Run P: 1+4x2+1 full passthrough stack (2 loops)" \
  env $COMMON $DATA RUN_ID=s4_P NUM_PRELUDE=1 NUM_SHARED=4 NUM_LOOPS=2 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 TIMESTEP_GAMMA_MAX=4.0 \
  USE_TIMESTEP_BIAS=1 USE_DEPTH_EMBED=1 USE_UNIQUE_NORMS=1 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

# --- Q: 1+4×3+1, full passthrough stack (depth embed + unique norms), 3 loops (compare vs s3_O) ---

run_experiment "Run Q: 1+4x3+1 full passthrough stack (3 loops)" \
  env $COMMON $DATA RUN_ID=s4_Q NUM_PRELUDE=1 NUM_SHARED=4 NUM_LOOPS=3 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 TIMESTEP_GAMMA_MAX=4.0 \
  USE_TIMESTEP_BIAS=1 USE_DEPTH_EMBED=1 USE_UNIQUE_NORMS=1 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

# --- R: 1+4×3+1, depth embed only (no unique norms), 3 loops (isolate depth embed, compare vs s3_O) ---

run_experiment "Run R: 1+4x3+1 depth embed only (3 loops)" \
  env $COMMON $DATA RUN_ID=s4_R NUM_PRELUDE=1 NUM_SHARED=4 NUM_LOOPS=3 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 TIMESTEP_GAMMA_MAX=4.0 \
  USE_TIMESTEP_BIAS=1 USE_DEPTH_EMBED=1 USE_UNIQUE_NORMS=0 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

# --- S: 1+4×3+1, unique norms only (no depth embed), 3 loops (isolate unique norms, compare vs s3_O) ---

run_experiment "Run S: 1+4x3+1 unique norms only (3 loops)" \
  env $COMMON $DATA RUN_ID=s4_S NUM_PRELUDE=1 NUM_SHARED=4 NUM_LOOPS=3 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 TIMESTEP_GAMMA_MAX=4.0 \
  USE_TIMESTEP_BIAS=1 USE_DEPTH_EMBED=0 USE_UNIQUE_NORMS=1 \
  torchrun --standalone --nproc_per_node=$NGPU $SCRIPT

echo ""
echo "==============================="
echo "  FULL-SCALE 3 SUMMARY"
echo "==============================="
echo "$SUMMARY"
echo "$FAILS run(s) failed."
