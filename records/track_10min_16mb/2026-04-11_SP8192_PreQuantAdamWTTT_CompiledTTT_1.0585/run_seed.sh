#!/bin/bash
# Documentation-only helper; bootstrap.sh launches seeds directly to keep the TensorPool entrypoint minimal.
set -euo pipefail
SEED="${SEED:?SEED must be set}"
exec torchrun --standalone --nproc_per_node=8 train_gpt.py
