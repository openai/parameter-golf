#!/usr/bin/env bash
# common_env.sh — Shared environment variables for all GDN architecture experiments.
# Source this file from experiment runners.
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
export DATA_DIR="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/data/tokenizers/fineweb_1024_bpe.model"
export VENV_PATH="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/.venv"

# ─── Training Hyperparameters ────────────────────────────────────────────────
export ITERATIONS=1000
export WARMDOWN_ITERS=500
export WARMUP_STEPS=10
export TRAIN_BATCH_TOKENS=131072
export TRAIN_SEQ_LEN=1024
export EVAL_SEQ_LEN=1024
export VAL_LOSS_EVERY=100
export SAVE_EVERY=500
export COMPILE_ENABLED=1

# ─── EMA / SWA / QAT ────────────────────────────────────────────────────────
export EMA_DECAY=0.997
export SWA_ENABLED=1
export SWA_EVERY=50
export LATE_QAT_THRESHOLD=0.15

# ─── Eval ────────────────────────────────────────────────────────────────────
export EVAL_STRIDE=64

# ─── GPU Detection & dtype ───────────────────────────────────────────────────
# V100 does not support bfloat16 and needs naive FLA kernels.
# A100+ uses bfloat16 (default in train_gdn_7k.py autocast).
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
if echo "$GPU_NAME" | grep -qi "V100"; then
    echo "[common_env] Detected V100 — enabling FLA_USE_NAIVE=1, TRAIN_DTYPE=float16"
    export FLA_USE_NAIVE=1
    export TRAIN_DTYPE=float16
else
    echo "[common_env] Detected GPU: $GPU_NAME — using bfloat16 defaults"
    export FLA_USE_NAIVE=0
    export TRAIN_DTYPE=bfloat16
fi

# ─── Data Subset Setup ───────────────────────────────────────────────────────
# Creates a data_subset/ directory in the experiment dir with symlinks to only
# the first 3 training shards and all available val shards.
# Sets DATA_PATH and SHARD_ORDER_FILE for the training script.
#
# Expects EXPERIMENT_DIR to be set by the caller before sourcing this file.
setup_data_subset() {
    local exp_dir="${1:?Usage: setup_data_subset <experiment_dir>}"
    local subset_dir="${exp_dir}/data_subset"
    mkdir -p "$subset_dir"

    # Link first 3 training shards
    local count=0
    for shard in "$DATA_DIR"/fineweb_train_*.bin; do
        [ -f "$shard" ] || continue
        ln -sf "$shard" "$subset_dir/$(basename "$shard")"
        count=$((count + 1))
        [ "$count" -ge 3 ] && break
    done
    echo "[common_env] Linked $count training shards into $subset_dir"

    # Link all val shards (only 1 exists currently)
    local vcount=0
    for shard in "$DATA_DIR"/fineweb_val_*.bin; do
        [ -f "$shard" ] || continue
        ln -sf "$shard" "$subset_dir/$(basename "$shard")"
        vcount=$((vcount + 1))
    done
    echo "[common_env] Linked $vcount validation shards into $subset_dir"

    # Create shard order file listing only the training shards
    local shard_order="${exp_dir}/shard_order.txt"
    ls "$subset_dir"/fineweb_train_*.bin 2>/dev/null | sort > "$shard_order"
    echo "[common_env] Shard order file: $shard_order ($(wc -l < "$shard_order") shards)"

    # Export paths for the training script
    export DATA_PATH="$subset_dir"
    export SHARD_ORDER_FILE="$shard_order"
}
