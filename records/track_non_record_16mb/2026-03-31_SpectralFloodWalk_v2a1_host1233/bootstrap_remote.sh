#!/usr/bin/env bash
set -euo pipefail

RECORD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RECORD_DIR}/../../.." && pwd)"
PYTHON_BIN="${SFW_PYTHON_BIN:-python3}"
TRAIN_SHARDS="${SFW_BOOTSTRAP_TRAIN_SHARDS:-80}"
SKIP_REQUIREMENTS="${SFW_BOOTSTRAP_SKIP_REQUIREMENTS:-0}"
SKIP_DATASET="${SFW_BOOTSTRAP_SKIP_DATASET:-0}"

cd "${REPO_ROOT}"

echo "[bootstrap] repo_root=${REPO_ROOT}"
echo "[bootstrap] python=${PYTHON_BIN}"
echo "[bootstrap] train_shards=${TRAIN_SHARDS}"

if [[ "${SKIP_REQUIREMENTS}" != "1" ]]; then
  echo "[bootstrap] checking Python deps"
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sentencepiece  # noqa: F401
import torch  # noqa: F401
PY
  then
    echo "[bootstrap] installing requirements.txt"
    "${PYTHON_BIN}" -m pip install -r requirements.txt
  else
    echo "[bootstrap] core deps already present"
  fi
else
  echo "[bootstrap] skipping requirements install"
fi

if [[ "${SKIP_DATASET}" != "1" ]]; then
  echo "[bootstrap] ensuring cached sp1024 dataset/tokenizer"
  "${PYTHON_BIN}" data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"
else
  echo "[bootstrap] skipping dataset fetch"
fi

echo "[bootstrap] running quick verification"
"${PYTHON_BIN}" -m py_compile \
  spectral_flood_walk_v2a_residual.py \
  spectral_flood_walk_v2a1_host1233.py \
  records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v2a1_host1233/train_gpt.py

"${PYTHON_BIN}" -m unittest tests.test_spectral_flood_walk_v2a_residual -v

echo "[bootstrap] ready"
echo "[bootstrap] next:"
echo "  cd ${RECORD_DIR}"
echo "  ./runpod_preflight.sh"
echo "  SFW_TARGET_GPU_COUNT=8 SFW_NPROC_PER_NODE=8 ./runpod_smoke.sh"
echo "  SFW_TARGET_GPU_COUNT=8 SFW_NPROC_PER_NODE=8 SFW_VAL_TOKEN_LIMIT=4194304 ./runpod_full.sh"
