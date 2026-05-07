#!/usr/bin/env bash
# SFT Random Feature Adapters — 8×H100 SXM leaderboard run
# Requires:  brotli sentencepiece torch-cuda flash-attn-3
#   pip install brotli sentencepiece
# Data download (run once):
#   MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
#   python3 data/cached_challenge_fineweb.py --variant sp8192

set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

SCRIPT="records/track_10min_16mb/2026-04-17_SFT_RandomFeatureAdapters/train_gpt.py"
export DATASETS_DIR="${DATASETS_DIR:-./data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_8192_bpe.model}"
export VOCAB_SIZE=8192
export TTT_ENABLED=1
export SEED="${SEED:-42}"

echo "=== SFT Random Feature Adapters  seed=${SEED} ==="
torchrun \
  --standalone \
  --nproc_per_node=8 \
  "$SCRIPT" \
  2>&1 | tee "train_seed${SEED}.log"

echo "=== Submission size ==="
wc -c final_model.int6.ptz
python3 -c "
sz = $(wc -c < final_model.int6.ptz)
code_sz = $(wc -c < $SCRIPT)
total = sz + code_sz
print(f'model: {sz:,} bytes')
print(f'code:  {code_sz:,} bytes')
print(f'total: {total:,} bytes  [{\"PASS\" if total <= 16_000_000 else \"FAIL\"}]')
"
