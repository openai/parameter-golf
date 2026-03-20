#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPO_ID="${REPO_ID:-willdepueoai/parameter-golf}"
REMOTE_ROOT="${REMOTE_ROOT:-datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/data/exports/tokenizer_ablation}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-$ROOT_DIR/data/tokenizer_specs.ablation.json}"

python3 data/download_hf_docs_and_tokenize.py \
  --repo-id "$REPO_ID" \
  --remote-root "$REMOTE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --tokenizer-config "$TOKENIZER_CONFIG"
