#!/bin/bash
# TTT ablations with wandb tracking — stride tests FIRST
# Uses saved pre-TTT model for fast iteration

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

echo "=== Starting TTT ablations (stride tests first) ==="

# 1. Per-window SGD stride=96 (FASTEST TO TEST — ~7 min)
echo "--- abl01: perwindow sgd stride96 bs128 ---"
RUN_ID=abl01_perwindow_sgd_s96_bs128 \
TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=0 EVAL_STRIDE=96 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -5
echo ""

# 2. Per-window SGD stride=128 (~5 min)
echo "--- abl02: perwindow sgd stride128 bs128 ---"
RUN_ID=abl02_perwindow_sgd_s128_bs128 \
TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=0 EVAL_STRIDE=128 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -5
echo ""

# 3. Per-window SGD stride=64 EATA on (~less than 10 min)
echo "--- abl03: perwindow sgd stride64 eata ---"
RUN_ID=abl03_perwindow_sgd_s64_eata \
TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=1 EVAL_STRIDE=64 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -5
echo ""

# 4. Per-window SGD stride=64 bs=128 (baseline ~10 min)
echo "--- abl04: perwindow sgd stride64 bs128 ---"
RUN_ID=abl04_perwindow_sgd_s64_bs128 \
TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=0 EVAL_STRIDE=64 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -5
echo ""

# 5. Chunked AdamW 1ep stride=64 (~2 min)
echo "--- abl05: chunked adamw 1ep stride64 ---"
RUN_ID=abl05_chunked_adamw_1ep_s64 \
TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.0005 \
TTT_EPOCHS=1 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 \
TTT_CHUNK_TOKENS=131072 EVAL_STRIDE=64 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -5
echo ""

# 6. Chunked AdamW 3ep stride=64 (~6 min)
echo "--- abl06: chunked adamw 3ep stride64 ---"
RUN_ID=abl06_chunked_adamw_3ep_s64 \
TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.0005 \
TTT_EPOCHS=3 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 \
TTT_CHUNK_TOKENS=131072 EVAL_STRIDE=64 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -5
echo ""

# 7. Chunked AdamW 3ep higher LR (~6 min)
echo "--- abl07: chunked adamw 3ep lr001 ---"
RUN_ID=abl07_chunked_adamw_3ep_lr001 \
TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.001 \
TTT_EPOCHS=3 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 \
TTT_CHUNK_TOKENS=131072 EVAL_STRIDE=64 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -5
echo ""

# 8. No TTT baseline (~2 min)
echo "--- abl08: no ttt ---"
RUN_ID=abl08_no_ttt \
TTT_ENABLED=0 EVAL_STRIDE=64 \
torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -5
echo ""

echo "=== ALL ABLATIONS COMPLETE ==="
