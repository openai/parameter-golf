#!/bin/bash
# === Parameter Golf: MoS Pilot on 1x H100 ===
# Paste this into your RunPod terminal.
# Total time: ~18 min download + 10 min MoS run = ~28 min
# Baseline already known: ~1.2244 bpb (10min/8xH100) or ~1.2074 (4hr/8xH100)

set -e

echo "=== Step 1: Download dataset ==="
# Run from the repo root (already cloned)
cd /workspace/parameter-golf

# Download full dataset (~18 min)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Verify dataset
echo "Train shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)"
echo "Val shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)"

echo ""
echo "=== Step 2: Run MoS K=2 rank=64 (10 min, 1x H100) ==="
echo "Start time: $(date)"

RUN_ID=mos_k2_r64_pilot \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=42 \
USE_MOS=1 \
MOS_K=2 \
MOS_RANK=64 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee /workspace/mos_log.txt

echo ""
echo "=== RESULTS ==="
echo ""
grep -E 'val_bpb|val_loss|bytes|param|model_params' /workspace/mos_log.txt | tail -15
echo ""
echo "Known baseline: val_bpb ~1.2244 (10min/8xH100)"
echo "Done at: $(date)"
