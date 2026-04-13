#!/usr/bin/env bash
# SwiGLU MLP experiment: enable gated MLP; optionally lower MLP_MULT for size budget.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
NPROC="${NPROC_PER_NODE:-8}"
MLP_SWIGLU=1 MLP_MULT="${MLP_MULT:-3.5}" torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN"
