#!/usr/bin/env bash
# Selective TTT: only blocks 3–5 (looped layers). Empty TTT_SELECTIVE_LAYERS = full model.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
NPROC="${NPROC_PER_NODE:-8}"
TTT_SELECTIVE_LAYERS="${TTT_SELECTIVE_LAYERS:-3,4,5}" torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN"
