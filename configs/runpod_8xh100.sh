#!/usr/bin/env bash
# =================================================================
# Parameter Golf — RunPod 8xH100 SXM Run Configurations
# =================================================================
# Usage (on RunPod pod):
#   source configs/runpod_8xh100.sh && rp_extended
#   source configs/runpod_8xh100.sh && rp_weekend
#   source configs/runpod_8xh100.sh && rp_with_ngram
#   source configs/runpod_8xh100.sh && rp_track_a_10min
#
# Throughput reference: 8xH100 SXM can push ~2-3M tok/s with 8-way
# DDP at 524K batch. These configs assume the CUDA train_gpt.py.
#
# Data: 71+ shards (7B+ tokens). Download with:
#   python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
# =================================================================

REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"

_rp_common() {
    export DATA_PATH="${REPO_DIR}/data/datasets/fineweb10B_sp1024"
    export TOKENIZER_PATH="${REPO_DIR}/data/tokenizers/fineweb_1024_bpe.model"
    export VOCAB_SIZE=1024
    export MAX_WALLCLOCK_SECONDS=0      # No wallclock cap
    export NCCL_IB_DISABLE=1

    # 8xH100: 524K tokens per step (65K per GPU)
    export TRAIN_BATCH_TOKENS=524288
    export TRAIN_SEQ_LEN=1024

    # Validation
    export VAL_BATCH_SIZE=524288
    export VAL_LOSS_EVERY=1000
    export EVAL_STRIDE=64

    # Logging
    export TRAIN_LOG_EVERY=50

    cd "$REPO_DIR"
}

_rp_launch() {
    torchrun --standalone --nproc_per_node=8 train_gpt.py
}

# -----------------------------------------------------------------
# rp_extended: 4000 iters x 524K batch = ~2.1B tokens, ~15 min
# -----------------------------------------------------------------
# Equivalent to local run_extended but finishes in minutes on H100s.
rp_extended() {
    _rp_common

    export ITERATIONS=4000
    export WARMUP_STEPS=30
    export WARMDOWN_ITERS=500
    export RUN_ID="rp_extended_$(date +%Y%m%d_%H%M%S)"

    echo "===== Parameter Golf — rp_extended (8xH100) ====="
    echo "Iters: 4000 | Batch: 524K | Total: ~2.1B tokens"
    echo "Estimated wall time: ~15 min @ 8xH100"
    echo "Run ID: $RUN_ID"
    echo "=================================================="

    _rp_launch
}

# -----------------------------------------------------------------
# rp_weekend: 10000 iters x 524K batch = ~5.2B tokens, ~40 min
# -----------------------------------------------------------------
# Equivalent to local run_weekend. Sees most of available data.
rp_weekend() {
    _rp_common

    export ITERATIONS=10000
    export WARMUP_STEPS=80
    export WARMDOWN_ITERS=1200
    export TRAIN_LOG_EVERY=100
    export RUN_ID="rp_weekend_$(date +%Y%m%d_%H%M%S)"

    echo "===== Parameter Golf — rp_weekend (8xH100) ====="
    echo "Iters: 10000 | Batch: 524K | Total: ~5.2B tokens"
    echo "Estimated wall time: ~40 min @ 8xH100"
    echo "Run ID: $RUN_ID"
    echo "================================================="

    _rp_launch
}

# -----------------------------------------------------------------
# rp_with_ngram: same as rp_extended + n-gram eval (Track B+)
# -----------------------------------------------------------------
rp_with_ngram() {
    _rp_common

    export ITERATIONS=4000
    export WARMUP_STEPS=30
    export WARMDOWN_ITERS=500
    export USE_NGRAM_EVAL=1
    export NGRAM_ORDER=9
    export RUN_ID="rp_ngram_$(date +%Y%m%d_%H%M%S)"

    echo "===== Parameter Golf — rp_with_ngram (8xH100, Track B+) ====="
    echo "Iters: 4000 | Batch: 524K | Total: ~2.1B tokens"
    echo "N-gram order: 9 | USE_NGRAM_EVAL: ON"
    echo "Run ID: $RUN_ID"
    echo "==============================================================="

    _rp_launch
}

# -----------------------------------------------------------------
# rp_track_a_10min: Official Track A — 10-min wallclock cap
# -----------------------------------------------------------------
# Maximizes iterations within the 10-minute limit.
# Uses high iteration count with wallclock cutoff.
rp_track_a_10min() {
    _rp_common

    export ITERATIONS=20000
    export MAX_WALLCLOCK_SECONDS=600
    export WARMUP_STEPS=20
    export WARMDOWN_ITERS=3500
    export RUN_ID="rp_track_a_$(date +%Y%m%d_%H%M%S)"

    echo "===== Parameter Golf — rp_track_a_10min (8xH100) ====="
    echo "Iters: 20000 (wallclock capped at 600s)"
    echo "Batch: 524K | Run ID: $RUN_ID"
    echo "======================================================="

    _rp_launch
}

echo "RunPod 8xH100 configs loaded. Available functions:"
echo "  rp_extended       — 4K iters, ~2.1B tokens, ~15min"
echo "  rp_weekend        — 10K iters, ~5.2B tokens, ~40min"
echo "  rp_with_ngram     — 4K iters + Track B+ n-gram eval"
echo "  rp_track_a_10min  — Official Track A (10-min wallclock)"
