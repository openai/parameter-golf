#!/bin/bash
# v6 smoke test — 60s, no TTT, no trigram, baseline compression
cd "/mnt/d/GitHub/Personal Projects/ParameterGolf/parameter-golf"
SCRIPT="records/track_10min_16mb/2026-03-22_pcloadloveletter_v6/train_gpt.py"
OUT="experiments/v6_smoke_test.txt"

echo "=== V6 SMOKE TEST ===" > $OUT
echo "Started: $(date)" >> $OUT

MAX_WALLCLOCK_SECONDS=60 TRAIN_BATCH_TOKENS=32768 VAL_LOSS_EVERY=50 \
TRAIN_LOG_EVERY=10 TTT_ENABLED=0 USE_TRIGRAM=0 USE_NOVEL_COMPRESSION=0 \
torchrun --standalone --nproc_per_node=1 $SCRIPT >> $OUT 2>&1

echo "" >> $OUT
echo "Finished: $(date)" >> $OUT
echo "V6_SMOKE_DONE" >> $OUT
