#!/usr/bin/env bash
# Full leaderboard training run (defaults match submission; ~10 min cap on 1xH100).
set -euo pipefail
RECORD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$RECORD_DIR/../../.." && pwd)"
cd "$ROOT"
export DATA_PATH="${DATA_PATH:-$ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export RUN_ID="${RUN_ID:-10L_Int5MLP_MuonWD04_SWA50}"
export SEED="${SEED:-42}"
NPROC="${NPROC:-1}"
exec torchrun --standalone --nproc_per_node="${NPROC}" "${RECORD_DIR}/train_gpt.py"
