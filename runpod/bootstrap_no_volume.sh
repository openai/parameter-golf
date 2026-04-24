#!/bin/bash
# Bootstrap for an 8xH100 AP-IN-1 pod (no Network Volume available in that DC).
# Everything lives on the container disk (ephemeral). Designed for a single
# sweep/record session — stop the pod after results are copied out.
#
# Expected inputs:
#   - HF_HOME env var pointing to /workspace/hf (already set by RunPod PG image)
#   - One of:
#       (a) HF_TOKEN env exported so we can pull the private processed SP8192
#           CaseOps shards from hf://FijaEE/parameter-golf-sp8192-caseops, OR
#       (b) Plain docs_selected.jsonl + our copy of prepare_caseops_data.py +
#           tokenizer (slower: ~45 min CPU on the 8xH100 box == $18 wasted).
#
# Prefer (a). See runpod/upload_caseops_to_hf.sh for how we publish the data.
#
# Usage on pod:
#   curl -sS https://raw.githubusercontent.com/Fija/parameter-golf/submission/pr1797-ngram-mix/runpod/bootstrap_no_volume.sh | bash
# or scp this file + exec.

set -euo pipefail

echo "[$(date)] bootstrap start"

# Install missing pip deps one-off (cheap on container disk — no volume needed).
pip install --quiet --break-system-packages brotli zstandard pyminify 2>&1 | tail -2 || true

cd /workspace
REPO=/workspace/parameter-golf
if [ ! -d "$REPO/.git" ]; then
  git clone --depth=1 --branch submission/pr1797-ngram-mix \
      https://github.com/Fija/parameter-golf.git "$REPO"
fi

cd "$REPO"

# Data: prefer the pre-tokenized HF dataset if available, else tokenize from docs_selected.jsonl.
DATA_DIR=/workspace/data
mkdir -p "$DATA_DIR/datasets" "$DATA_DIR/tokenizers"

# Redirect all caches to container disk (40 GB — plenty for this session).
export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_inductor
export TRITON_CACHE_DIR=/workspace/triton
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"

HF_DATASET_PRETOK="${HF_DATASET_PRETOK:-FijaEE/parameter-golf-sp8192-caseops}"

echo "[$(date)] attempting to pull pre-tokenized shards from $HF_DATASET_PRETOK"
python3 - <<PY || TOKENIZE_FROM_SCRATCH=1
import os, sys
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(
        repo_id=os.environ["HF_DATASET_PRETOK"],
        repo_type="dataset",
        local_dir="/workspace/data/datasets/fineweb10B_sp8192_caseops",
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )
    print("pre-tok download ok:", p)
except Exception as e:
    print("pre-tok fetch failed:", e, file=sys.stderr)
    sys.exit(1)
PY

if [ "${TOKENIZE_FROM_SCRATCH:-0}" = "1" ]; then
  echo "[$(date)] pre-tok unavailable — tokenizing from scratch (~45 min CPU)"
  cd "$REPO"
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 2>&1 | tail -3
  # cached_challenge_fineweb puts docs_selected.jsonl into ./data/datasets/
  python3 records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix/prepare_caseops_data.py \
      --docs data/datasets/docs_selected.jsonl \
      --out  $DATA_DIR/datasets/fineweb10B_sp8192_caseops/datasets \
      --sp   records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
fi

echo "[$(date)] bootstrap done. data at:"
du -sh $DATA_DIR/datasets/* 2>/dev/null | head -5
