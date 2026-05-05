#!/bin/bash
# Extra ablations: original per-window SGD with various batch sizes

BASE_ENV="export PATH=/data/backups/rganapa/pylibs/bin:\$PATH \
TMPDIR=/data/backups/rganapa/tmp \
PYTHONPATH=/data/backups/rganapa/pylibs \
TRITON_CACHE_DIR=/data/backups/rganapa/triton_cache \
TORCH_HOME=/data/backups/rganapa/torch_home \
WANDB_DIR=/data/backups/rganapa \
WANDB_API_KEY=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 \
WANDB_PROJECT=parameter-golf \
DATA_PATH=/data/backups/rganapa/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/backups/rganapa/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=14 BIGRAM_VOCAB_SIZE=8192 BIGRAM_DIM=64 \
MLP_ACTIVATION=leaky2 ROPE_BASE=50000 EVAL_STRIDE=64 \
EVAL_ONLY_MODEL=/data/backups/rganapa/parameter-golf/final_model_pre_ttt.pt \
SEED=1337"

CD="cd /data/backups/rganapa/parameter-golf"
SCRIPT="clean_train_201_eata_ttt.py"

run_ablation() {
    local name=$1
    shift
    local extra_vars="$@"
    echo "========================================="
    echo "STARTING: $name"
    echo "========================================="
    eval $BASE_ENV $extra_vars && $CD && \
        RUN_ID="abl_${name}" torchrun --nproc_per_node=8 $SCRIPT 2>&1 | \
        grep -E "online_ttt|perwindow|eval_only|val_bpb|eval_time|ttt_progress|done"
    echo "FINISHED: $name"
    echo "========================================="
    echo ""
}

# Original per-window SGD with EATA off, various batch sizes
# This is what got 1.1126 before

# batch_seqs=128 (what we already ran)
run_ablation "perwindow_sgd_bs128" \
    "TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=0"

# batch_seqs=192
run_ablation "perwindow_sgd_bs192" \
    "TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=192 TTT_EATA=0"

# batch_seqs=256
run_ablation "perwindow_sgd_bs256" \
    "TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=256 TTT_EATA=0"

# Per-window SGD with EATA on (skip easy batches)
run_ablation "perwindow_sgd_bs128_eata" \
    "TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=1"

# Per-window AdamW (instead of SGD)
run_ablation "perwindow_adamw_bs128" \
    "TTT_ENABLED=1 TTT_MODE=perwindow TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.0005 TTT_EPOCHS=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=0"

# --- Stride ablations with original per-window SGD ---

# stride=96 (33% fewer windows)
run_ablation "perwindow_sgd_bs128_stride96" \
    "TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=0 EVAL_STRIDE=96"

# stride=128 (50% fewer windows)
run_ablation "perwindow_sgd_bs128_stride128" \
    "TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_EATA=0 EVAL_STRIDE=128"

echo "ALL EXTRA ABLATIONS COMPLETE"
