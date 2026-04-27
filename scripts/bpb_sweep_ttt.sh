#!/usr/bin/env bash
# Phase 1b: TTT sweep (epochs, chunk, lr). Edit loops or export BASE_ENV vars.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
NPROC="${NPROC_PER_NODE:-8}"
# Default competition-style TTT; override as needed
export TTT_ENABLED="${TTT_ENABLED:-1}"
for EPOCHS in 3 4 5; do
  for CHUNK in 32768 16384; do
    for TLR in 0.003 0.005 0.008; do
      echo "========== TTT_EPOCHS=$EPOCHS TTT_CHUNK_TOKENS=$CHUNK TTT_LR=$TLR =========="
      TTT_EPOCHS="$EPOCHS" TTT_CHUNK_TOKENS="$CHUNK" TTT_LR="$TLR" \
        torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN" || true
    done
  done
done
