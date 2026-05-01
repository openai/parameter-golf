#!/usr/bin/env bash
set -euo pipefail

cd /home/frosty40/sota_rascal

PYTHON_BIN="${PYTHON_BIN:-python3}"
DOCS_PATH="${DOCS_PATH:-/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl}"
OUT_ROOT="${OUT_ROOT:-/home/frosty40/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets}"
MAX_TRAIN_SHARDS="${MAX_TRAIN_SHARDS:-80}"
VAL_DOCS="${VAL_DOCS:-50000}"
SHARD_TOKENS="${SHARD_TOKENS:-10000000}"
TOKENIZER_SKIP_DOCS="${TOKENIZER_SKIP_DOCS:-50000}"

args=(
  scripts/prepare_sp10240_caseops_data.py
  --docs "${DOCS_PATH}"
  --out "${OUT_ROOT}"
  --train-tokenizer
  --val-docs "${VAL_DOCS}"
  --max-train-shards "${MAX_TRAIN_SHARDS}"
  --shard-tokens "${SHARD_TOKENS}"
  --tokenizer-skip-docs "${TOKENIZER_SKIP_DOCS}"
)

if [[ -n "${TOKENIZER_TRAIN_DOCS:-}" ]]; then
  args+=(--tokenizer-train-docs "${TOKENIZER_TRAIN_DOCS}")
fi

echo "[sp10240-caseops] start $(date -Iseconds)"
echo "[sp10240-caseops] docs=${DOCS_PATH}"
echo "[sp10240-caseops] out=${OUT_ROOT}"
echo "[sp10240-caseops] max_train_shards=${MAX_TRAIN_SHARDS} val_docs=${VAL_DOCS} shard_tokens=${SHARD_TOKENS}"
exec "${PYTHON_BIN}" "${args[@]}"
