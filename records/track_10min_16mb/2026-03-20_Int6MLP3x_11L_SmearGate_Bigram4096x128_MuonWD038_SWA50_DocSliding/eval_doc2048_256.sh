#!/usr/bin/env bash

set -euo pipefail

RECORD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RECORD_DIR/../../.." && pwd)"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

cd "$RECORD_DIR"

CHECKPOINT_PATH="${1:-$RECORD_DIR/final_model.pt}"
SUMMARY_OUT="${2:-$RECORD_DIR/eval_doc2048_256.csv}"

python checkpoint_frontier_sweep.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --summary-out "$SUMMARY_OUT" \
  --data-path "${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}" \
  --tokenizer-path "${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}" \
  --num-layers 11 \
  --mlp-mult 3 \
  --smeargate-enabled \
  --bigram-vocab-size 4096 \
  --bigram-dim 128 \
  --train-seq-len 2048 \
  --eval-seq-lens 2048 \
  --strides 256 \
  --modes doc_sliding \
  --variant-names prequant,int6_zstd_core
