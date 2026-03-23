#!/bin/bash
# v6 Official 8xH100 Scoring Run
# Must use novel compression (int6+zstd produces 18+ MB, over 16 MB cap)
# TTT: 10 epochs AdamW with cosine schedule + per-layer lr
# Trigram: OFF for first run (tight on size), test with ON in second run

cd /workspace/parameter-golf  # RunPod workspace path

echo "=== V6 OFFICIAL 8xH100 RUN ===" | tee -a v6_run.log
echo "Started: $(date)" | tee -a v6_run.log

# Download data if not present
if [ ! -d data/datasets/fineweb10B_sp1024 ]; then
    echo "Downloading data..." | tee -a v6_run.log
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
fi

# RUN 1: Full scoring run — novel compression + TTT
echo "" | tee -a v6_run.log
echo "=== RUN 1: SCORING (novel compression + TTT 10ep) ===" | tee -a v6_run.log
TTT_ENABLED=1 TTT_EPOCHS=10 TTT_BATCH_SEQS=16 \
USE_TRIGRAM=0 USE_NOVEL_COMPRESSION=1 \
EVAL_STRIDE=32 EVAL_BATCH_SEQS=64 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-22_pcloadloveletter_v6/train_gpt.py \
  2>&1 | tee -a v6_run.log

echo "" | tee -a v6_run.log
echo "Finished: $(date)" | tee -a v6_run.log
echo "V6_RUN_DONE" | tee -a v6_run.log
