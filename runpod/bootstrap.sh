#!/bin/bash
# RunPod pod-entry bootstrap. Meant to live on a persistent Network Volume
# mounted at /workspace. First-time setup below; subsequent pod spins skip the
# clone / data download entirely.
#
# Layout expected on the Network Volume (/workspace/):
#   cache/
#     torch_inductor/    — torch.compile kernel cache (saves 90–120s/run)
#     triton/            — Triton kernel cache
#     hf/                — HuggingFace datasets cache
#   data/
#     datasets/fineweb10B_sp8192/   — pre-tokenized shards (~20 GB, populated once)
#     datasets/fineweb10B_sp1024/   — legacy variant if ever needed
#     tokenizers/                   — SentencePiece .model files
#   repo/parameter-golf/            — git clone of the competition repo
#   runs/<run_id>/                  — training logs + quantized artifacts
#
# Usage:
#   # one-off per pod (or put in RunPod Template > Container Start Command):
#   /workspace/bootstrap.sh
#   cd /workspace/repo/parameter-golf && git pull --rebase
#   /workspace/launch_experiment.sh <submission_folder> <seed>

set -euo pipefail

WS=/workspace
mkdir -p "$WS/cache/torch_inductor" "$WS/cache/triton" "$WS/cache/hf"
mkdir -p "$WS/data/datasets" "$WS/data/tokenizers" "$WS/runs"

# Redirect compile caches to volume. Saves ~2 min on warm spin-ups.
export TORCHINDUCTOR_CACHE_DIR=$WS/cache/torch_inductor
export TRITON_CACHE_DIR=$WS/cache/triton
export HF_HOME=$WS/cache/hf
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

# One-time repo clone. If already there, just pull.
REPO=$WS/repo/parameter-golf
if [ ! -d "$REPO/.git" ]; then
  mkdir -p "$WS/repo"
  git clone --depth=1 https://github.com/openai/parameter-golf.git "$REPO"
fi

# One-time dataset download if not already on volume.
DS=$WS/data/datasets/fineweb10B_sp8192
if [ ! -f "$DS/fineweb_val_000000.bin" ]; then
  echo "[bootstrap] downloading FineWeb SP8192 shards (one-time, ~20 GB)..."
  cd "$REPO"
  python3 data/cached_challenge_fineweb.py --variant sp8192 \
      --output-dir "$WS/data/datasets/fineweb10B_sp8192" \
      --tokenizer-dir "$WS/data/tokenizers"
fi

# Symlink the volume data into the repo checkout for the standard env vars.
cd "$REPO"
if [ ! -L ./data ] && [ ! -d ./data ]; then
  ln -s "$WS/data" ./data
elif [ -d ./data ] && [ ! -L ./data ]; then
  # Merge repo's data/ (contains python helpers) with volume data/ via bind mount
  # — safest is to not touch it if it already exists as a directory.
  rsync -a "$WS/data/" ./data/
fi

echo "[bootstrap] ready. Example run:"
echo "  cd $REPO && git pull --rebase"
echo "  /workspace/launch_experiment.sh records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix 42"
