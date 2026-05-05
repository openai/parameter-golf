#!/bin/bash
# Test AdamW + cosine LR for per-window TTT on WD=0.05+QEP model
# Eval-only using saved final_model_pre_ttt.pt

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

# Common config
export EMA_ENABLED=1
export EMA_DECAY=0.997
export NUM_LAYERS=14
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
export MLP_ACTIVATION=leaky2
export TTT_ENABLED=1
export TTT_MODE=perwindow
export TTT_EPOCHS=1
export TTT_BATCH_SEQS=128
export PRUNE_FRAC=0.0
export ROPE_BASE=50000
export SWA_ENABLED=0
export GPTQ_ENABLED=1
export GPTQ_SAMPLES=256
export QEP_ENABLED=1
export SEED=1337
export EVAL_ONLY_MODEL=/data/backups/rganapa/parameter-golf/final_model_pre_ttt.pt

cd /data/backups/rganapa/parameter-golf

# Test 1: AdamW + cosine, lr=0.0001, stride=76 (baseline comparison)
echo "=== AdamW lr=0.0001 cosine stride=76 ==="
export TTT_OPTIMIZER=adamw
export TTT_ADAMW_LR=0.0001
export TTT_COSINE_LR=1
export TTT_FREEZE_LAYERS=2
export EVAL_STRIDE=76
export WANDB_RUN_NAME=adamw_ttt_lr0001_s76
torchrun --nproc_per_node=8 clean_train_201_eata_ttt.py
echo "=== AdamW lr=0.0001 s76 DONE ==="

# Test 2: AdamW + cosine, lr=0.0005 (201 script default), stride=76
echo "=== AdamW lr=0.0005 cosine stride=76 ==="
export TTT_ADAMW_LR=0.0005
export WANDB_RUN_NAME=adamw_ttt_lr0005_s76
torchrun --nproc_per_node=8 clean_train_201_eata_ttt.py
echo "=== AdamW lr=0.0005 s76 DONE ==="

# Test 3: SGD baseline for comparison (same as exp206)
echo "=== SGD lr=0.002 stride=76 (baseline) ==="
export TTT_OPTIMIZER=sgd
export TTT_LR=0.002
export TTT_MOMENTUM=0.9
export TTT_COSINE_LR=0
export WANDB_RUN_NAME=sgd_ttt_baseline_s76
torchrun --nproc_per_node=8 clean_train_201_eata_ttt.py
echo "=== SGD baseline DONE ==="
