#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash starter_kit/scripts/01_runpod_bootstrap.sh https://github.com/YOUR_GITHUB_USERNAME/parameter-golf.git

FORK_URL="${1:-}"
if [[ -z "$FORK_URL" ]]; then
  echo "Provide your fork URL as first arg."
  exit 1
fi

cd /workspace
if [[ ! -d parameter-golf ]]; then
  git clone "$FORK_URL" parameter-golf
fi

cd parameter-golf
git remote -v

echo "Downloading small dataset slice for low-cost iteration..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

echo "Bootstrap complete. Run: bash starter_kit/scripts/02_smoke_run.sh"
