#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TARGET_DIR="$REPO_ROOT/records/track_10min_16mb/2026-04-20_SP8192_LegalTTT_W104_FaithfulReplay"
LOG_PATH="/workspace/w104_seed314.log"
VENV_PATH="/workspace/venv"

if [[ ! -d "$VENV_PATH" ]]; then
  python3 -m venv "$VENV_PATH" --system-site-packages
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt" huggingface_hub datasets sentencepiece

cd "$REPO_ROOT"
MATCHED_FINEWEB_REPO_ID="kevclark/parameter-golf" \
  python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80

cd "$TARGET_DIR"
SEED=314 python train_gpt.py 2>&1 | tee "$LOG_PATH"

python - <<'PY'
import re
from pathlib import Path
log_path = Path('/workspace/w104_seed314.log')
text = log_path.read_text(encoding='utf-8', errors='replace')
matches = re.findall(r'quantized_ttt[^\n]*val_bpb\s*[:=]\s*([0-9]+\.[0-9]+)', text)
if not matches:
    matches = re.findall(r'val_bpb\s*[:=]\s*([0-9]+\.[0-9]+)', text)
if not matches:
    raise SystemExit('Could not find val_bpb in log')
print(f'final quantized_ttt val_bpb: {matches[-1]}')
PY
