#!/bin/bash
cd /workspace/parameter-golf

# Test 1: baseline sanity check (no memory tokens)
echo "=== BASELINE CHECK ==="
RUN_ID=baseline_check DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 ITERATIONS=10 WARMUP_STEPS=2 MAX_WALLCLOCK_SECONDS=60 VAL_LOSS_EVERY=0 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tail -20
