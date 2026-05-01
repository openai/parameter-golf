#!/usr/bin/env bash
set -euo pipefail

PERSIST_ROOT="${PERSIST_ROOT:-/workspace}"
REPO_REMOTE="${REPO_REMOTE:-https://github.com/IanniMuliterno/parameter-golf.git}"
REPO_DIR="${REPO_DIR:-$PERSIST_ROOT/parameter-golf}"
VENV_DIR="${VENV_DIR:-$PERSIST_ROOT/.venvs/parameter-golf-cu128-torch291}"
SP_VARIANT="${SP_VARIANT:-sp8192}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-kevclark/parameter-golf}"
FLASH_ATTN_WHEEL_INDEX="${FLASH_ATTN_WHEEL_INDEX:-https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/}"
RECORD_DIR_REL="records/track_10min_16mb/2026-04-20_SP8192_3LayerRecur_ParResid_QK525_AttnGate_MP4TTT"

mkdir -p "$PERSIST_ROOT"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$REPO_REMOTE" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch --depth 1 origin
  current_branch="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"
  git -C "$REPO_DIR" pull --ff-only origin "$current_branch"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_DIR/requirements.txt"
python -m pip install -r "$REPO_DIR/$RECORD_DIR_REL/requirements.txt"

if ! python - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("flash_attn_interface") is not None else 1)
PY
then
  python -m pip install flash_attn_3 --no-deps --find-links "$FLASH_ATTN_WHEEL_INDEX"
fi

dataset_dir="$REPO_DIR/data/datasets/fineweb10B_${SP_VARIANT}"
tokenizer_path="$REPO_DIR/data/tokenizers/fineweb_${SP_VARIANT#sp}_bpe.model"
train_probe="$(printf "%s/fineweb_train_%06d.bin" "$dataset_dir" $((TRAIN_SHARDS - 1)))"
val_probe="$dataset_dir/fineweb_val_000000.bin"

if [[ ! -f "$tokenizer_path" || ! -f "$val_probe" || ! -f "$train_probe" ]]; then
  MATCHED_FINEWEB_REPO_ID="$MATCHED_FINEWEB_REPO_ID" \
    python "$REPO_DIR/data/cached_challenge_fineweb.py" --variant "$SP_VARIANT" --train-shards "$TRAIN_SHARDS"
fi

cat <<EOF
Bootstrap complete.

Reusable paths:
  repo: $REPO_DIR
  venv: $VENV_DIR
  data: $REPO_DIR/data

Run this on every new pod after attaching the same persistent volume at $PERSIST_ROOT:
  source "$VENV_DIR/bin/activate"
  cd "$REPO_DIR/$RECORD_DIR_REL"

Priority run check:
  python - <<'PY'
import flash_attn_interface
print("flash_attn_interface: OK")
PY
EOF
