#!/bin/bash
# Freeze sweep on WD=0.05 + QEP model (eval-only, uses saved final_model_pre_ttt.pt)
# Tests freeze=0/4/6/8 at stride=64

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
export TTT_LR=0.002
export TTT_EPOCHS=1
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=128
export PRUNE_FRAC=0.0
export ROPE_BASE=50000
export SWA_ENABLED=0
export GPTQ_ENABLED=1
export GPTQ_SAMPLES=256
export QEP_ENABLED=1
export SEED=1337
export EVAL_STRIDE=64

cd /data/backups/rganapa/parameter-golf

for FREEZE in 0 4 6 8; do
    echo "=== FREEZE=${FREEZE} stride=64 ==="
    export TTT_FREEZE_LAYERS=$FREEZE
    export WANDB_RUN_NAME="freeze_sweep_wd05_f${FREEZE}_s64"
    export EVAL_ONLY_MODEL=/data/backups/rganapa/parameter-golf/final_model_pre_ttt.pt
    torchrun --nproc_per_node=8 clean_train_201_eata_ttt.py
    echo "=== FREEZE=${FREEZE} DONE ==="
    echo ""
done
