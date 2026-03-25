#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SCRIPT_DIR"

python3 -m pip install --quiet --upgrade pip setuptools wheel packaging ninja
python3 -m pip install --quiet -r requirements.txt
python3 -m pip install --quiet --no-build-isolation flash-attn
python3 "$REPO_ROOT/data/cached_challenge_fineweb.py" --variant sp1024
bash "$SCRIPT_DIR/run_official_8x_h100.sh"
