#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://huggingface.co/datasets/Frosty40/10k_golfer/resolve/main"
DATA_ROOT="${DATA_ROOT:-/workspace/SOTA_FINAL/data}"
DATASET_DIR="$DATA_ROOT/datasets/fineweb10B_sp10240_first80"
TOKENIZER_DIR="$DATA_ROOT/tokenizers"

mkdir -p "$DATASET_DIR" "$TOKENIZER_DIR"

download_file() {
  local name="$1"
  local dest="$2"
  if [[ -s "$dest" ]]; then
    echo "exists $dest"
    return
  fi
  echo "download $name -> $dest"
  curl -fL --retry 8 --retry-all-errors --connect-timeout 20 -C - \
    -o "$dest.part" "$BASE_URL/$name?download=true"
  mv "$dest.part" "$dest"
}

download_file "fineweb_10240_bpe.model" "$TOKENIZER_DIR/fineweb_10240_bpe.model"
download_file "fineweb_10240_bpe.vocab" "$TOKENIZER_DIR/fineweb_10240_bpe.vocab"

for idx in $(seq 0 79); do
  shard="$(printf 'fineweb_train_%06d.bin' "$idx")"
  download_file "$shard" "$DATASET_DIR/$shard"
done

download_file "fineweb_val_000000.bin" "$DATASET_DIR/fineweb_val_000000.bin"

train_count="$(find "$DATASET_DIR" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l)"
val_count="$(find "$DATASET_DIR" -maxdepth 1 -name 'fineweb_val_*.bin' | wc -l)"
echo "train_count=$train_count val_count=$val_count"
test "$train_count" -eq 80
test "$val_count" -eq 1
test -f "$TOKENIZER_DIR/fineweb_10240_bpe.model"
