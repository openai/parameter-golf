#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at $PYTHON_BIN" >&2
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/diffusion_week3_scale_long.env}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/logs/week3_stage_h_continue_20260412_200615/diffusion_week3_scale_diffusion_best_mlx.npz}"
OUT_DIR="${OUT_DIR:-$(dirname "$CHECKPOINT_PATH")}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-0}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

set -a
source "$CONFIG_PATH"
set +a

export OUT_DIR="$OUT_DIR"
export VAL_MAX_TOKENS="$VAL_MAX_TOKENS"

exec "$PYTHON_BIN" diffusion_eval.py --checkpoint "$CHECKPOINT_PATH"
