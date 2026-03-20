#!/bin/bash
# === Parameter Golf: MoS 1-Hour Validation on 1x H100 ===
# Usage: bash setup_and_run_1h.sh
# The script runs training inside nohup so it survives terminal disconnects.
# Log is written to /workspace/mos_1h_log.txt — check with: tail -f /workspace/mos_1h_log.txt

set -e

echo "=== Step 1: Download dataset ==="
cd /workspace/parameter-golf

# HF token for faster downloads
export HF_TOKEN="${HF_TOKEN:-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}"

# Download full dataset (~18 min, skips if already present)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Verify dataset
TRAIN_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "Train shards: $TRAIN_COUNT  Val shards: $VAL_COUNT"
if [ "$TRAIN_COUNT" -lt 1 ]; then
    echo "ERROR: No training shards found. Dataset download failed."
    exit 1
fi

echo ""
echo "=== Step 2: Run MoS K=2 R=64 (1 HOUR, 1x H100) ==="
echo "Start time: $(date)"
echo ""
echo "Training will run in the background via nohup."
echo "Monitor with:  tail -f /workspace/mos_1h_log.txt"
echo "Check GPU with: nvidia-smi"
echo "Safe to close terminal — training will continue."
echo ""

nohup bash -c '
RUN_ID=mos_k2_r64_1h \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=42 \
USE_MOS=1 \
MOS_K=2 \
MOS_RANK=64 \
WARMDOWN_ITERS=100 \
MAX_WALLCLOCK_SECONDS=3600 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
' > /workspace/mos_1h_log.txt 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
echo "PID saved to /workspace/train.pid"
echo "$TRAIN_PID" > /workspace/train.pid

# Wait a few seconds and confirm it started
sleep 5
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "Training is running. You can safely close this terminal."
    echo ""
    echo "=== Quick commands ==="
    echo "  Monitor:  tail -f /workspace/mos_1h_log.txt"
    echo "  Status:   nvidia-smi"
    echo "  Kill:     kill \$(cat /workspace/train.pid)"
else
    echo "ERROR: Training process died. Check /workspace/mos_1h_log.txt"
    tail -20 /workspace/mos_1h_log.txt
    exit 1
fi
