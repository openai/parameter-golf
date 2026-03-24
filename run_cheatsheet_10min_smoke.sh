#!/usr/bin/env bash
# Smoke run (~10 min wall clock): reaches phase-1 → transition → stabilization → phase 2
# on a typical single GPU. Tune ITERATIONS / TRAIN_BATCH_TOKENS if you OOM or finish too fast.
set -euo pipefail
cd "$(dirname "$0")"

export RUN_ID="${RUN_ID:-cheatsheet_10min_smoke}"
# 800 iters → phase1 @ 20% = step 160; stab 2% = 16 steps; rest is phase 2 (fits in 600s on most GPUs)
export ITERATIONS="${ITERATIONS:-800}"
export PHASE1_FRAC="${PHASE1_FRAC:-0.20}"
export STABILIZATION_FRAC="${STABILIZATION_FRAC:-0.02}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-50}"
export MAX_SUBMISSION_BYTES="${MAX_SUBMISSION_BYTES:-0}"

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

torchrun --standalone --nproc_per_node=1 train_gpt_cheatsheet.py
