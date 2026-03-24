#!/bin/bash
# Test TTT@coarse + rescore@s64 strategy
# Uses saved pre-TTT model (eval-only mode)
#
# Strategy: TTT adapts weights at coarse stride (fast), then rescore at s64 (precise)
# Expected: ~1.1099 BPP in ~440-550s total

export PATH=/data/backups/rganapa/pylibs/bin:$PATH
export TMPDIR=/data/backups/rganapa/tmp
export PYTHONPATH=/data/backups/rganapa/pylibs
export TRITON_CACHE_DIR=/data/backups/rganapa/triton_cache
export TORCH_HOME=/data/backups/rganapa/torch_home
export WANDB_DIR=/data/backups/rganapa
export WANDB_API_KEY=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
export WANDB_PROJECT=parameter-golf
export DATA_PATH=/data/backups/rganapa/parameter-golf/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/data/backups/rganapa/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
export NUM_LAYERS=14
export BIGRAM_VOCAB_SIZE=8192
export BIGRAM_DIM=64
export MLP_ACTIVATION=leaky2
export ROPE_BASE=50000
export EVAL_ONLY_MODEL=/data/backups/rganapa/parameter-golf/final_model_pre_ttt.pt
export SEED=1337

cd /data/backups/rganapa/parameter-golf
SCRIPT=clean_train_201_eata_ttt.py

echo "=== TTT@coarse + rescore@s64 strategy tests — $(date) ==="

# NOTE: The 201 script doesn't support separate TTT stride vs scoring stride yet.
# For now, we test TTT at various strides — the s64 rescore happens automatically
# as the "second sliding window" eval in the script.

# Test 1: TTT at stride=96 (458s TTT + ~91s rescore = ~549s total)
echo "--- TTT@s96 + rescore@s64 ---"
RUN_ID=rescore_ttt96_s64 \
TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=0 EVAL_STRIDE=96 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -8
echo "DONE TTT@s96"
echo ""

# Test 2: TTT at stride=128 (347s TTT + ~91s rescore = ~438s total)
echo "--- TTT@s128 + rescore@s64 ---"
RUN_ID=rescore_ttt128_s64 \
TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=0 EVAL_STRIDE=128 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -8
echo "DONE TTT@s128"
echo ""

# Test 3: TTT at stride=192 (very coarse, ~230s est + 91s = ~321s)
echo "--- TTT@s192 + rescore@s64 ---"
RUN_ID=rescore_ttt192_s64 \
TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=0 EVAL_STRIDE=192 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -8
echo "DONE TTT@s192"
echo ""

echo "=== RESCORE STRATEGY COMPLETE — $(date) ==="
