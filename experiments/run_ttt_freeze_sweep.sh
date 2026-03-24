#!/bin/bash
# TTT freeze sweep: test different numbers of frozen layers
# Uses saved pre-TTT model (eval-only), stride=64 to see if we can get back to best BPB

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

echo "=== TTT Freeze Sweep — $(date) ==="

# All use stride=64 (best BPB) to see if faster TTT lets us fit in 600s
for FREEZE in 2 4 6 8 10; do
    echo "--- freeze=${FREEZE} layers, stride=64 ---"
    RUN_ID=freeze_sweep_f${FREEZE}_s64 \
    TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
    TTT_FREEZE_LAYERS=$FREEZE TTT_BATCH_SEQS=128 TTT_EATA=0 EVAL_STRIDE=64 \
    torchrun --nproc_per_node=8 $SCRIPT 2>&1 | tail -5
    echo "DONE freeze=$FREEZE"
    echo ""
done

echo "=== FREEZE SWEEP COMPLETE — $(date) ==="
