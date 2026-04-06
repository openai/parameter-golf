#!/usr/bin/env bash
# Full 20k-step training run — respects SCRIPT, RUN_ID and all hyperparameter env vars.
#
# Usage:
#   bash run_baseline.sh                                      # plain baseline
#   SCRIPT=train_gpt_stack.py bash run_baseline.sh           # run the stack
#   RUN_ID=my_run SCRIPT=train_gpt_stack.py bash run_baseline.sh
#   MDL_LAMBDA=0.1 SCRIPT=train_gpt_stack.py bash run_baseline.sh
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

: "${SCRIPT:=train_gpt.py}"
: "${RUN_ID:=baseline_$(basename ${SCRIPT%.py})}"

mkdir -p logs

echo "========================================"
echo "  SCRIPT : $SCRIPT"
echo "  RUN_ID : $RUN_ID"
echo "  Log    : logs/${RUN_ID}.log"
echo "========================================"

RUN_ID="$RUN_ID" \
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}" \
torchrun --standalone --nproc_per_node=1 "$SCRIPT" 2>&1 | tee "logs/${RUN_ID}.log"

echo ""
echo "Done. Final BPB:"
grep "val_bpb:" "logs/${RUN_ID}.log" | tail -5
