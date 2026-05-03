#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
export DATA_DIR=/data/parameter-golf/data
export MAX_WALLCLOCK_SECONDS=600
export PGOLF_PYTHON=/data/pgolf_venv/bin/python
export PYTHONUNBUFFERED=1
export QUANT_CLIP_ANNEAL_ALPHA=0.08
export QUANT_CLIP_ANNEAL_FULL_AT=0.92
export QUANT_CLIP_ANNEAL_START=0.55
export QUANT_EMBED_ANNEAL_ALPHA=0.03
export RUN_ID=pr1394-h4-quant-clip
export SEED=1337
export TRAIN_LOG_EVERY=500
export VAL_LOSS_EVERY=4000
export VOCAB_SIZE=8192
set +e
${PGOLF_PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt_human.py > results/train.log 2>&1
rc=$?
set -e
printf '{"returncode": %s, "finished_at": "%s"}\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > results/runner_status.json
exit $rc
