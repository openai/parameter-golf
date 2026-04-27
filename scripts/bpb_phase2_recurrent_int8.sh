#!/usr/bin/env bash
# Mixed GPTQ: int8 for weights in LOOP_START..LOOP_END; rest uses MATRIX_BITS (int6).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
NPROC="${NPROC_PER_NODE:-8}"
GPTQ_RECURRENT_INT8=1 torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN"
