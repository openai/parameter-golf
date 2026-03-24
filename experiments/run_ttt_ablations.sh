#!/bin/bash
# TTT ablations using saved pre-TTT model (no retraining needed)
# Each run takes ~2-5 min

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
    echo "Extra vars: $extra_vars"
    echo "========================================="
    eval $BASE_ENV $extra_vars && $CD && \
        RUN_ID="abl_${name}" torchrun --nproc_per_node=8 $SCRIPT 2>&1 | \
        grep -E "chunked_ttt|perwindow|eval_only|val_bpb|eval_time|ttt_progress|done"
    echo ""
    echo "FINISHED: $name"
    echo "========================================="
    echo ""
}

# --- Chunked AdamW ablations ---

# Baseline: 1 epoch, AdamW, lr=0.0005, chunk=131K (already ran: 1.1175, 138s)
run_ablation "chunked_adamw_1ep_lr0005" \
    "TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.0005 TTT_EPOCHS=1 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_CHUNK_TOKENS=131072"

# 3 epochs, AdamW, lr=0.0005, chunk=131K
run_ablation "chunked_adamw_3ep_lr0005" \
    "TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.0005 TTT_EPOCHS=3 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_CHUNK_TOKENS=131072"

# 1 epoch, higher LR=0.001
run_ablation "chunked_adamw_1ep_lr001" \
    "TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.001 TTT_EPOCHS=1 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_CHUNK_TOKENS=131072"

# 3 epochs, higher LR=0.001
run_ablation "chunked_adamw_3ep_lr001" \
    "TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.001 TTT_EPOCHS=3 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_CHUNK_TOKENS=131072"

# 1 epoch, smaller chunks (65K) for more frequent adaptation
run_ablation "chunked_adamw_1ep_chunk65k" \
    "TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.0005 TTT_EPOCHS=1 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_CHUNK_TOKENS=65536"

# 3 epochs, smaller chunks (65K)
run_ablation "chunked_adamw_3ep_chunk65k" \
    "TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.0005 TTT_EPOCHS=3 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_CHUNK_TOKENS=65536"

# No cosine LR (flat)
run_ablation "chunked_adamw_3ep_flat_lr" \
    "TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.0005 TTT_EPOCHS=3 TTT_COSINE_LR=0 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_CHUNK_TOKENS=131072"

# Freeze 0 layers (all unfrozen)
run_ablation "chunked_adamw_3ep_freeze0" \
    "TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=adamw TTT_ADAMW_LR=0.0005 TTT_EPOCHS=3 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=0 TTT_BATCH_SEQS=128 TTT_CHUNK_TOKENS=131072"

# --- Chunked SGD ablations ---

# 3 epochs SGD for comparison
run_ablation "chunked_sgd_3ep" \
    "TTT_ENABLED=1 TTT_MODE=chunked TTT_OPTIMIZER=sgd TTT_LR=0.002 TTT_EPOCHS=3 TTT_COSINE_LR=1 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 TTT_CHUNK_TOKENS=131072 TTT_MOMENTUM=0.9"

# --- No TTT baseline ---
run_ablation "no_ttt" \
    "TTT_ENABLED=0 EVAL_STRIDE=64"

echo "ALL ABLATIONS COMPLETE"
