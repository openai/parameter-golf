#!/usr/bin/env bash
# Gate: RASCAL_III_SLOT_F — 1-GPU, 2000 steps. Run BEFORE the 8x run.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SEED="${SEED:-444}"

PYTHONPATH_EXTRA=""
if [[ -d "${REPO_ROOT}/flash-attention/hopper" ]]; then
    PYTHONPATH_EXTRA="${REPO_ROOT}/flash-attention/hopper:"
fi

SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=2000 \
SKIP_GPTQ=1 \
SKIP_FINAL_EVAL=1 \
LOADER_MODE=coprime \
COPRIME_MAX_LOADED_SHARDS=1 \
COPRIME_SHARDS_PER_BATCH=1 \
COPRIME_SHARD_HOLD_STEPS=64 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
NGRAM_EVAL_ORDER=0 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=0 \
SLOT_ENABLED=1 \
PACK_INT6_6BIT=1 \
EVAL_STRIDE=64 \
POST_EMA_DIAGNOSTIC=1 \
PYTHONPATH="${PYTHONPATH_EXTRA}${PYTHONPATH:-}" \
torchrun --standalone --nproc_per_node=1 "${SCRIPT_DIR}/train_gpt_slot.py" \
2>&1 | tee "${SCRIPT_DIR}/gate_seed${SEED}.log"

echo "--- gate done. check step_avg and loss trend before proceeding to run.sh ---"
