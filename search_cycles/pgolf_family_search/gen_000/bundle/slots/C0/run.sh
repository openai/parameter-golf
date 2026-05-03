#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
export DATA_PATH=/data/parameter-golf/data/datasets/fineweb10B_sp1024
export MAX_WALLCLOCK_SECONDS=180
export PGOLF_PYTHON=/data/pgolf_venv/bin/python
export PYTHONUNBUFFERED=1
export TOKENIZER_PATH=/data/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
export TRAIN_LOG_EVERY=50
export VAL_LOSS_EVERY=200
set +e
$PGOLF_PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py > results/train.log 2>&1
rc=$?
set -e
printf '{"returncode": %s, "finished_at": "%s"}\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > results/runner_status.json
exit $rc
