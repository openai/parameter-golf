#!/usr/bin/env bash
set -euo pipefail

# Download dataset if not present
if [ ! -f ./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin ]; then
    echo "Downloading dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

echo "Starting training on DGX Spark (single GPU, Blackwell sm_121)"

RUN_ID=${RUN_ID:-spark_baseline} \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-200} \
TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-10} \
torchrun --standalone --nproc_per_node=1 train_gpt_spark.py

# Auto-register completed run in history
python3 -c "
import run_tracker, os
run_id = os.environ.get('RUN_ID', 'unknown')
log_path = f'logs/{run_id}.txt'
parsed = run_tracker.parse_log(log_path)
if parsed.get('final_bpb') is not None:
    params = {}
    for key in ['NUM_LAYERS','MODEL_DIM','MLP_MULT','MATRIX_LR','SCALAR_LR',
                'TIED_EMBED_LR','MUON_MOMENTUM','WARMDOWN_ITERS','SEED',
                'ITERATIONS','MAX_WALLCLOCK_SECONDS','TRAIN_SEQ_LEN']:
        val = os.environ.get(key)
        if val is not None:
            try: params[key] = float(val) if '.' in val else int(val)
            except ValueError: params[key] = val
    duration = parsed['all_train_points'][-1]['train_time_ms']/1000 if parsed['all_train_points'] else 0
    run_tracker.save_run(run_id=run_id, params_dict=params,
        final_bpb=parsed['final_bpb'], final_loss=parsed['final_loss'],
        steps_completed=parsed['steps'], duration_seconds=round(duration,1),
        status='completed', artifact_size_bytes=parsed.get('artifact_size_bytes'))
    print(f'Saved {run_id}: bpb={parsed[\"final_bpb\"]:.4f}')
" || true
