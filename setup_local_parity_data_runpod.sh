#!/bin/bash
set -euo pipefail

SRC_ROOT="${1:-/workspace/parameter-golf/data}"
DEST_ROOT="${2:-/tmp/parameter-golf-data}"
VARIANT="${3:-fineweb10B_sp1024}"
TOKENIZER_BASENAME="${4:-fineweb_1024_bpe}"

SRC_DATASET_DIR="$SRC_ROOT/datasets/$VARIANT"
SRC_TOKENIZER_DIR="$SRC_ROOT/tokenizers"
DEST_DATASET_DIR="$DEST_ROOT/datasets/$VARIANT"
DEST_TOKENIZER_DIR="$DEST_ROOT/tokenizers"

mkdir -p "$DEST_DATASET_DIR" "$DEST_TOKENIZER_DIR"

echo "staging dataset from $SRC_DATASET_DIR to $DEST_DATASET_DIR"
cp -a "$SRC_DATASET_DIR"/. "$DEST_DATASET_DIR"/

echo "staging tokenizer from $SRC_TOKENIZER_DIR to $DEST_TOKENIZER_DIR"
cp -a "$SRC_TOKENIZER_DIR/$TOKENIZER_BASENAME.model" "$DEST_TOKENIZER_DIR/"
if [ -f "$SRC_TOKENIZER_DIR/$TOKENIZER_BASENAME.vocab" ]; then
  cp -a "$SRC_TOKENIZER_DIR/$TOKENIZER_BASENAME.vocab" "$DEST_TOKENIZER_DIR/"
fi

echo "local parity data ready"
echo "DATA_PATH=$DEST_ROOT/datasets/$VARIANT"
echo "TOKENIZER_PATH=$DEST_ROOT/tokenizers/$TOKENIZER_BASENAME.model"
echo "dataset_files=$(find "$DEST_DATASET_DIR" -maxdepth 1 -type f | wc -l)"
echo "dataset_bytes=$(du -sh "$DEST_DATASET_DIR" | cut -f1)"
