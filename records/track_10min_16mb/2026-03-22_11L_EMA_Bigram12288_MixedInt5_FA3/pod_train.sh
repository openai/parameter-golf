#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/dev/shm/parameter-golf-local/simon-l1-r20}"
NPROC="${NPROC_PER_NODE:-8}"
PYTHON_BIN="${POD_PYTHON_BIN:-/workspace/parameter-golf/.venv/bin/python}"

cd "$ROOT"
exec env SIMON_ENV_FILE=.env "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" train_gpt.py
