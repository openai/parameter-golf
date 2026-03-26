#!/bin/bash
# Round 1: Run PR #834 baseline on all 8 GPUs independently (single GPU each)
# Each GPU runs the full pipeline: train + eval
# This establishes the baseline BPP for each GPU

export PATH=/data/backups/rganapa/pylibs/bin:$PATH
export PYTHONPATH=/data/backups/rganapa/pylibs
export TMPDIR=/data/backups/rganapa/tmp
export TRITON_CACHE_DIR=/data/backups/rganapa/triton_cache
export TORCH_HOME=/data/backups/rganapa/torch_home
export DATA_PATH=data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model
export TTT_CHUNK_TOKENS=1048576
export TTT_EPOCHS=4
export TTT_LR=0.0005
export TTT_FREEZE_BLOCKS=2
export MAX_WALLCLOCK_SECONDS=600
export TRAIN_BATCH_TOKENS=98304
export PYTHONUNBUFFERED=1

cd /data/backups/rganapa/parameter-golf
mkdir -p eight_parallel_logs

for GPU in 0 1 2 3 4 5 6 7; do
    echo "=== Launching GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU \
    SEED=$((1337 + GPU)) \
    RUN_ID=round1_gpu${GPU} \
    WANDB_DISABLED=true \
    nohup python3 pr834_train_gpt.py > eight_parallel_logs/round1_gpu${GPU}.log 2>&1 &
    echo "PID=$!"
done

echo "All 8 launched. Check eight_parallel_logs/round1_gpu*.log"
