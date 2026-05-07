#!/bin/bash
# Generate SP16384 tokenizer and dataset for larger vocab experiment.
# Requires access to the HuggingFace docs dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Generating SP16384 tokenizer ==="

cat > "${REPO_ROOT}/data/tokenizer_specs_16384.json" << 'EOF'
[
  {
    "name": "sp_bpe_16384",
    "kind": "sentencepiece_bpe",
    "vocab_size": 16384,
    "tokenizer_train_docs": 5000000
  }
]
EOF

cd "$REPO_ROOT"
python3 data/download_hf_docs_and_tokenize.py \
    --output-root data \
    --tokenizer-config data/tokenizer_specs_16384.json \
    --skip-byte

echo "=== SP16384 tokenizer and dataset generated ==="
echo "Run training with: VOCAB_SIZE=16384 EMBED_BITS=5 EMBED_CLIP_SIGMAS=10.0"
