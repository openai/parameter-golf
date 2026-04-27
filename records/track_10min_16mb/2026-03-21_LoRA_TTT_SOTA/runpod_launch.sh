#!/usr/bin/env bash
# RunPod launch script for LoRA TTT SOTA experiment
# SSH into your pod, then run: bash runpod_launch.sh
# For 8xH100 leaderboard run. For 1xH100 testing, change nproc_per_node=1

set -e

# ── 1. Setup (only needed once per pod) ───────────────────────────────────────
if [ ! -d "/workspace/parameter-golf" ]; then
    echo "=== Cloning repo ==="
    cd /workspace
    git clone https://github.com/openai/parameter-golf.git
    cd parameter-golf

    echo "=== Downloading FineWeb data ==="
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo "=== Repo already exists, skipping setup ==="
    cd /workspace/parameter-golf
fi

# ── 2. Copy experiment files ───────────────────────────────────────────────────
RECORD_DIR="records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA"
mkdir -p "$RECORD_DIR"

# Copy the experiment script (you've synced your local repo or use rsync)
# If running from local machine: rsync -avz --exclude='.git' . runpod:/workspace/parameter-golf/
echo "=== Using train_gpt.py from $RECORD_DIR ==="

# ── 3. Run training + eval ─────────────────────────────────────────────────────
cd "/workspace/parameter-golf/$RECORD_DIR"

echo ""
echo "=== Experiment: LoRA TTT on SOTA ==="
echo "=== Training: identical to SOTA 1.1428 ==="
echo "=== Eval: LoRA TTT rank=8, stride=256 ==="
echo ""

# Default: 3 seeds for statistical significance
for SEED in 42 1337 2024; do
    echo "--- Seed $SEED ---"
    SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
        2>&1 | tee "train_seed${SEED}.log"
    echo "--- Seed $SEED complete ---"
done

echo ""
echo "=== All runs complete. Check train_seed*.log for results. ==="
echo "=== Look for: ttt_eval_exact val_bpb=... ==="
