#!/bin/bash
# Recursive transformer: wider model to use the headroom
# 7 unique blocks × 2 reps = 14L, but at dim=640 instead of 512
# More capacity per layer, should improve BPP while still fitting in 16MB

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
export EMA_ENABLED=1
export EMA_DECAY=0.997
export NUM_LAYERS=14
export MODEL_DIM=640
export NUM_HEADS=10
export NUM_KV_HEADS=5
export BIGRAM_VOCAB_SIZE=8192
export BIGRAM_DIM=64
export MUON_WD=0.05
export ADAM_WD=0.02
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3500
export ITERATIONS=9000
export EVAL_STRIDE=76
export MLP_ACTIVATION=leaky2
export TTT_ENABLED=1
export TTT_MODE=perwindow
export TTT_LR=0.002
export TTT_EPOCHS=1
export TTT_MOMENTUM=0.9
export TTT_FREEZE_LAYERS=2
export TTT_BATCH_SEQS=128
export PRUNE_FRAC=0.0
export ROPE_BASE=50000
export SWA_ENABLED=0
export GPTQ_ENABLED=1
export GPTQ_SAMPLES=256
export QEP_ENABLED=1
export INT5_MLP=0
export SEED=1337
export N_UNIQUE_BLOCKS=7

cd /data/backups/rganapa/parameter-golf

echo "=== Recursive 7-unique dim=640 WD=0.05 ==="
export WANDB_RUN_NAME=recursive_7ub_dim640_wd05
torchrun --nproc_per_node=8 clean_train_211_recursive.py
echo "=== Recursive 7-unique dim=640 DONE ==="
