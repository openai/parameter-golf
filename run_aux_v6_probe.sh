#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="/workspace/venv"
PREFERRED_REPO="/workspace/parameter-golf-openai"
FALLBACK_REPO="/workspace/parameter-golf"
AUX_DATASET="8Planetterraforming/Parameter-Golf-V6-Privacy-Web-Filtering"

if [[ ! -d "$VENV_PATH" ]]; then
  python3 -m venv "$VENV_PATH" --system-site-packages
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

if [[ -d "$PREFERRED_REPO" ]]; then
  REPO_PATH="$PREFERRED_REPO"
elif [[ -d "$FALLBACK_REPO" ]]; then
  REPO_PATH="$FALLBACK_REPO"
  echo "WARNING: $PREFERRED_REPO not found; using $FALLBACK_REPO"
else
  echo "ERROR: Neither $PREFERRED_REPO nor $FALLBACK_REPO exists."
  exit 1
fi

cd "$REPO_PATH"

if [[ ! -f "data/cached_challenge_fineweb.py" ]]; then
  echo "ERROR: Missing helper script data/cached_challenge_fineweb.py"
  exit 1
fi

echo "Installing/validating dependencies in $VENV_PATH ..."
pip install huggingface_hub datasets || echo "WARNING: pip install failed (likely network/proxy); relying on already-available packages if present."
python3 - <<'PY'
import huggingface_hub
import datasets
print('huggingface_hub import OK:', huggingface_hub.__version__)
print('datasets import OK:', datasets.__version__)
PY

echo "Caching official FineWeb challenge data (variant=sp8192) ..."
python3 data/cached_challenge_fineweb.py --variant sp8192 || echo "WARNING: FineWeb caching failed (check HF/network access)."

echo "Downloading auxiliary dataset: $AUX_DATASET ..."
python3 - <<'PY'
import os
from datasets import load_dataset_builder
from huggingface_hub import snapshot_download

name = "8Planetterraforming/Parameter-Golf-V6-Privacy-Web-Filtering"
path = ""
errors = []

try:
    path = snapshot_download(repo_id=name, repo_type="dataset")
except Exception as e:
    errors.append(f"snapshot_download failed: {e}")

if not path:
    try:
        builder = load_dataset_builder(name)
        path = builder.cache_dir
    except Exception as e:
        errors.append(f"load_dataset_builder failed: {e}")

if path:
    print(f"Aux dataset local path: {os.path.abspath(path)}")
else:
    print("Aux dataset local path: <not downloaded>")
    for err in errors:
        print("WARNING:", err)
PY

echo "WARNING: FineWeb remains the main training corpus for Parameter Golf; auxiliary V6 should be mixed conservatively (start at 1%)."
echo "Probe setup complete. Full multi-seed record training was NOT started."
