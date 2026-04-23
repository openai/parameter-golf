#!/usr/bin/env bash
# ============================================================
# RunPod setup for OpenAI Parameter Golf
# Run this once after SSH into your RunPod pod.
# ============================================================
set -e

REPO_DIR="/workspace/parameter-golf"
VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"

echo "==> Cloning repo..."
if [ ! -d "$REPO_DIR" ]; then
  cd /workspace
  git clone https://github.com/openai/parameter-golf.git
fi
cd "$REPO_DIR"

echo "==> Pulling latest..."
git pull

echo "==> Installing Python deps..."
pip install -r requirements.txt -q

echo "==> Downloading FineWeb dataset (variant=$VARIANT, shards=$TRAIN_SHARDS)..."
python3 data/cached_challenge_fineweb.py --variant "$VARIANT" --train-shards "$TRAIN_SHARDS"

echo ""
echo "==> Setup complete. To launch training:"
echo ""
echo "  # 1xH100 baseline:"
echo "  RUN_ID=baseline_sp1024 \\"
echo "  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \\"
echo "  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\"
echo "  VOCAB_SIZE=1024 \\"
echo "  torchrun --standalone --nproc_per_node=1 train_gpt.py"
echo ""
echo "  # 8xH100 (full leaderboard run):"
echo "  RUN_ID=sota_8xh100 \\"
echo "  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \\"
echo "  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\"
echo "  VOCAB_SIZE=1024 \\"
echo "  torchrun --standalone --nproc_per_node=8 train_gpt.py"
echo ""
echo "  # Run the current SOTA config (best model: 1.0810 bpb):"
echo "  cd records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
echo "  RUN_ID=sota_repro \\"
echo "  DATA_PATH=../../../data/datasets/fineweb10B_sp8192/ \\"
echo "  TOKENIZER_PATH=../../../data/tokenizers/fineweb_8192_spm.model \\"
echo "  VOCAB_SIZE=8192 \\"
echo "  torchrun --standalone --nproc_per_node=8 train_gpt.py"
