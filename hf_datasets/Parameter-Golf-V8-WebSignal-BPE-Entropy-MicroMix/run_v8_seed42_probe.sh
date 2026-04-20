#!/usr/bin/env bash
set -euo pipefail

HF_DATASET_REPO="8Planetterraforming/Parameter-Golf-V8-WebSignal-BPE-Entropy-MicroMix"

DATASET_DIR="$({ python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "8Planetterraforming/Parameter-Golf-V8-WebSignal-BPE-Entropy-MicroMix"
path = snapshot_download(repo_id=repo_id, repo_type="dataset")
print(path)
PY
} | tail -n 1)"

export DATASET_DIR

echo "Using dataset snapshot: $DATASET_DIR"
python "$DATASET_DIR/build_v8_micro_mix.py"

echo
echo "Recommended micro-mix rates (unchanged): 0.02%, 0.05%, 0.10%"
echo "Pass condition (unchanged): seed42 must beat 1.08041364 before 3-seed proof"
