#!/bin/bash
# v6 TTT test — 60s training, then 2 epochs of AdamW TTT, eval stride=256 (fast)
cd "/mnt/d/GitHub/Personal Projects/ParameterGolf/parameter-golf"
SCRIPT="records/track_10min_16mb/2026-03-22_pcloadloveletter_v6/train_gpt.py"
OUT="experiments/v6_ttt_test.txt"

echo "=== V6 TTT TEST ===" > $OUT
echo "Started: $(date)" >> $OUT

MAX_WALLCLOCK_SECONDS=60 TRAIN_BATCH_TOKENS=32768 VAL_LOSS_EVERY=50 \
TRAIN_LOG_EVERY=10 TTT_ENABLED=1 TTT_EPOCHS=2 TTT_BATCH_SEQS=8 \
USE_TRIGRAM=0 USE_NOVEL_COMPRESSION=0 EVAL_STRIDE=256 \
torchrun --standalone --nproc_per_node=1 $SCRIPT >> $OUT 2>&1

echo "" >> $OUT
echo "Finished: $(date)" >> $OUT
echo "V6_TTT_DONE" >> $OUT
