#!/usr/bin/env bash
# =================================================================
# Parameter Golf — Local Run Configurations (M3 Ultra 96GB)
# =================================================================
# Usage:
#   source configs/local_runs.sh && run_extended
#   source configs/local_runs.sh && run_weekend
#   source configs/local_runs.sh && run_with_ngram
#
# Throughput baseline: ~44K tok/s with 65K batch on M3 Ultra 96GB.
# Data available: 71+ shards (7B+ tokens).
# =================================================================

_PG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_local_common() {
    # Shared settings for all local runs
    export DATA_PATH="${_PG_ROOT}/data/datasets/fineweb10B_sp1024"
    export TOKENIZER_PATH="${_PG_ROOT}/data/tokenizers/fineweb_1024_bpe.model"
    export VOCAB_SIZE=1024
    export MLX_EAGER_EVAL=1
    export MLX_MAX_MICROBATCH_TOKENS=32768
    export MAX_WALLCLOCK_SECONDS=0      # No wallclock cap for local

    # Batch size: 65536 tokens per step, no grad accum
    export TRAIN_BATCH_TOKENS=65536
    export GRAD_ACCUM_STEPS=1

    # Validation
    export VAL_BATCH_SIZE=524288
    export VAL_LOSS_EVERY=0             # Eval at end only (saves time)

    # Logging
    export TRAIN_LOG_EVERY=200
}

# -----------------------------------------------------------------
# run_extended: 30K iters x 65K batch = ~2B tokens, ~12 hours
# -----------------------------------------------------------------
# Good for overnight or day-long runs. Sees ~28% of available data.
# Warmdown covers last ~12% of training (3500 iters).
run_extended() {
    _local_common

    export ITERATIONS=30000
    export WARMUP_STEPS=30
    export WARMDOWN_ITERS=3500
    export RUN_ID="local_extended_$(date +%Y%m%d_%H%M%S)"

    echo "===== Parameter Golf — run_extended ====="
    echo "Iters: 30000 | Batch: 65K | Total: ~1.97B tokens"
    echo "Estimated wall time: ~12 hours @ 44K tok/s"
    echo "Run ID: $RUN_ID"
    echo "=========================================="

    cd "$_PG_ROOT"
    .venv/bin/python -u train_gpt_mlx.py
}

# -----------------------------------------------------------------
# run_weekend: 80K iters x 65K batch = ~5.2B tokens, ~32 hours
# -----------------------------------------------------------------
# Full weekend run. Sees ~74% of available data (71 shards).
# Extended warmdown (8000 iters, ~10%) for better final loss.
run_weekend() {
    _local_common

    export ITERATIONS=80000
    export WARMUP_STEPS=80
    export WARMDOWN_ITERS=8000
    export TRAIN_LOG_EVERY=500
    export RUN_ID="local_weekend_$(date +%Y%m%d_%H%M%S)"

    echo "===== Parameter Golf — run_weekend ====="
    echo "Iters: 80000 | Batch: 65K | Total: ~5.24B tokens"
    echo "Estimated wall time: ~33 hours @ 44K tok/s"
    echo "Run ID: $RUN_ID"
    echo "========================================="

    cd "$_PG_ROOT"
    .venv/bin/python -u train_gpt_mlx.py
}

# -----------------------------------------------------------------
# run_with_ngram: same as run_extended + n-gram eval (Track B+)
# -----------------------------------------------------------------
# Adds USE_NGRAM_EVAL=1 for Track B+ scoring at the end of training.
# N-gram eval adds ~5-10 min overhead at completion.
run_with_ngram() {
    _local_common

    export ITERATIONS=30000
    export WARMUP_STEPS=30
    export WARMDOWN_ITERS=3500
    export USE_NGRAM_EVAL=1
    export NGRAM_ORDER=9
    export RUN_ID="local_ngram_$(date +%Y%m%d_%H%M%S)"

    echo "===== Parameter Golf — run_with_ngram (Track B+) ====="
    echo "Iters: 30000 | Batch: 65K | Total: ~1.97B tokens"
    echo "Estimated wall time: ~12 hours + ngram eval @ end"
    echo "N-gram order: 9 | USE_NGRAM_EVAL: ON"
    echo "Run ID: $RUN_ID"
    echo "======================================================"

    cd "$_PG_ROOT"
    .venv/bin/python -u train_gpt_mlx.py
}

echo "Local run configs loaded. Available functions:"
echo "  run_extended    — 30K iters, ~2B tokens, ~12h"
echo "  run_weekend     — 80K iters, ~5.2B tokens, ~33h"
echo "  run_with_ngram  — 30K iters + Track B+ n-gram eval"
