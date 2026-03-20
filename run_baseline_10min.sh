#!/bin/bash
# Vanilla baseline, 10 min, 1x H100. Survives terminal disconnect.
# Monitor: tail -f /workspace/baseline_10min_log.txt

cd /workspace/parameter-golf

nohup bash -c '
RUN_ID=baseline_10min \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=42 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
' > /workspace/baseline_10min_log.txt 2>&1 &

echo "PID: $!"
echo "Monitor: tail -f /workspace/baseline_10min_log.txt"
