#!/bin/bash
# F1 v1 — Accuracy profile: correction head + distillation
set -euo pipefail

SEED="${SEED:-1337}" \
F1_CORR_RANK=256 \
F1_CORR_SCALE_INIT=0.10 \
DISTILL_ENABLED=1 \
DISTILL_STEPS=24 \
DISTILL_LR_FACTOR=0.02 \
DISTILL_TEMPERATURE=1.5 \
DISTILL_ALPHA=0.60 \
DISTILL_KL_CLIP=10.0 \
bash "$(dirname "${BASH_SOURCE[0]}")/run.sh"
