#!/bin/bash
# Experiment 1: SOTA + TTT 4 epochs only (safe baseline, validated by PR #1812)
# Run this first — fastest to confirm we beat 1.0810

set -e
SCRIPT="experiments/train_gpt_sota.py"
LOG_DIR="logs/exp1_ttt4"
mkdir -p $LOG_DIR

for SEED in 42 1337 2024; do
  echo "=== Exp1 SEED=$SEED ==="
  TTT_ENABLED=1 QK_GAIN_INIT=5.25 TTT_EPOCHS=4 SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 $SCRIPT \
    2>&1 | tee $LOG_DIR/seed${SEED}.log
  echo "val_bpb seed $SEED: $(grep 'val_bpb' $LOG_DIR/seed${SEED}.log | tail -1)"
done

echo "=== Exp1 results ==="
grep 'val_bpb' $LOG_DIR/*.log | grep -v step
