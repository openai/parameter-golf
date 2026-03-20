#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install mlx

echo
echo "Bootstrap complete."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  ./scripts/verify_env.sh"
