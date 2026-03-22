#!/usr/bin/env bash
# Download FineWeb shards + tokenizer into ./data/ (run from anywhere).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
exec python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"
