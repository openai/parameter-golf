#!/usr/bin/env bash
# Bigram hash embedding (~65K extra params classically); tune BIGRAM_DIM if needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
NPROC="${NPROC_PER_NODE:-8}"
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1024}" BIGRAM_DIM="${BIGRAM_DIM:-128}" \
  torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN"
