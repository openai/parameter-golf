#!/usr/bin/env bash
set -euo pipefail

if [[ -f environment/vars.env ]]; then
  set -a
  source environment/vars.env
  set +a
fi

: "${SEED:=1337}"
: "${MAX_WALLCLOCK_SECONDS:=600}"

torchrun --standalone --nproc_per_node=8 train_gpt.py
