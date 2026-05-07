#!/usr/bin/env bash
# 3-seed runner for the Pre-Quant TTT + Val-Calib GPTQ + SLOT-24 quad-stack synthesis.
# Run this from the repo root after data download. Each seed: ~10 min train + ~9 min eval = ~19 min wall.
# Total wallclock for 3 seeds: ~60 min on 8xH100 SXM (~$3-5 per seed on RunPod).

set -euo pipefail

# Resolve script's own folder so we can write logs next to the script
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Sanity: train_gpt.py must exist next to this script
if [ ! -f "train_gpt.py" ]; then
  echo "ERROR: train_gpt.py not found in $SCRIPT_DIR" >&2
  exit 1
fi

# Repo root has the data/ folder. We need DATA_DIR to point at it.
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/}"

if [ ! -d "$DATA_DIR/datasets/fineweb10B_sp8192" ]; then
  echo "ERROR: SP8192 dataset not found at $DATA_DIR/datasets/fineweb10B_sp8192" >&2
  echo "       Run: MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192" >&2
  exit 1
fi

# Hyperparameters for the synthesis. These match the README's expected gain table.
export VOCAB_SIZE=8192

# Pre-Quant TTT (Track A) — pushed harder than PR #1487
export PREQUANT_TTT_ENABLED=1
export PREQUANT_TTT_EPOCHS=11
export PREQUANT_TTT_FREEZE_BLOCKS=0
export PREQUANT_TTT_LR=0.00050
export PREQUANT_TTT_COSINE_DECAY=1

# Val-Calibrated GPTQ — Hessians computed on validation data
export GPTQ_CALIB_SOURCE=val

# SLOT-24 — per-window hidden delta + logit bias on the frozen post-quant model
# Replaces eval-time legal TTT in this synthesis (much bigger gain per eval second)
export SLOT_ENABLED=1
export SLOT_STEPS=24
export SLOT_LR=0.012
export SLOT_LR_MIN=0.001
export SLOT_BATCH_SEQS=32
export SLOT_EVAL_STRIDE=96

# Eval-Time Legal Score-First TTT — disabled by default (SLOT supersedes it)
# Set TTT_ENABLED=1 SLOT_ENABLED=0 to use this fallback path
export TTT_ENABLED=0
export TTT_LR=0.005
export TTT_EPOCHS=2
export TTT_FREEZE_BLOCKS=2
export TTT_CHUNK_TOKENS=32768
export TTT_MOMENTUM=0.9

# Architecture knobs (same as PR #1487 plus QK gain bump)
export QK_GAIN_INIT=5.5
export RECUR_LAYERS="3,4,5"
export RECUR_START_STEP=3000
export PARALLEL_START_LAYER=7
export EMA_DECAY=0.9965

# Run all 3 seeds for statistical significance
for SEED in 42 1337 2024; do
  echo "============================================"
  echo "=== Synthesis seed=$SEED  GPUs=8 ==="
  echo "============================================"
  RUN_ID="synthesis_seed${SEED}" \
  SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "train_seed${SEED}.log"
  echo "=== seed=$SEED done ==="
done

# Print the final per-seed numbers for quick review
echo ""
echo "============ FINAL VAL_BPB BY SEED ============"
for SEED in 42 1337 2024; do
  echo "--- seed $SEED ---"
  grep -E "(final_int6_sliding_window|final_int6_slot|final_int6_ttt|post-prequant-ttt|val_calib_gptq|slot_eval:done)" "train_seed${SEED}.log" || true
done
echo "==============================================="
