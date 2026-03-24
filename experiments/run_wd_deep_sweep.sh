#!/bin/bash
# Deep WD sweep: testing below 0.05 since lower keeps winning
# WD=0.05 got 1.1058. Does it keep going?

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
export SWA_ENABLED=0
export EMA_ENABLED=1
export EMA_DECAY=0.997
export ADAM_WD=0.02
export WARMDOWN_ITERS=3500
export ITERATIONS=9000
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export GPTQ_ENABLED=1
export GPTQ_SAMPLES=256
export PRUNE_FRAC=0.0
export SEED=1337
export TTT_ENABLED=1
export TTT_MODE=perwindow
export TTT_LR=0.002
export TTT_EPOCHS=1
export TTT_MOMENTUM=0.9
export TTT_FREEZE_LAYERS=2
export TTT_BATCH_SEQS=128
export EVAL_STRIDE=76

cd /data/backups/rganapa/parameter-golf

echo "=== Deep WD Sweep — $(date) ==="

for WD in 0.03 0.04 0.045; do
    echo "--- WD=$WD ---"
    RUN_ID=wd_deep_${WD} MUON_WD=$WD \
    torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -5
    echo "DONE WD=$WD"
    echo ""
done

echo "=== DEEP WD SWEEP COMPLETE — $(date) ==="
