#!/bin/bash
# Experiment 2: Polar Express NS + TTT 4 epochs (our new contribution)

set -e
SCRIPT="records/track_10min_16mb/2026-03-20_PolarExpress4TTT/train_gpt.py"
LOG_DIR="logs/exp2_pe_ttt4"
mkdir -p $LOG_DIR

for SEED in 42 1337 2024; do
  echo "=== Exp2 SEED=$SEED ==="
  TTT_ENABLED=1 QK_GAIN_INIT=5.25 TTT_EPOCHS=4 SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 $SCRIPT \
    2>&1 | tee $LOG_DIR/seed${SEED}.log
  echo "val_bpb seed $SEED: $(grep 'val_bpb' $LOG_DIR/seed${SEED}.log | tail -1)"
done

echo "=== Exp2 results ==="
grep 'val_bpb' $LOG_DIR/*.log | grep -v step
