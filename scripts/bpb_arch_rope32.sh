#!/usr/bin/env bash
# Richer RoPE: ROPE_DIMS=32 (baseline often 16).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
NPROC="${NPROC_PER_NODE:-8}"
ROPE_DIMS=32 torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN"
