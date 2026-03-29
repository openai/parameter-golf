#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python evaluator/bootstrap_skydiscover_records.py \
  --output-dir skydiscover_bootstrap_smoke \
  --checkpoint-name checkpoint_0 \
  --normalize-evaluator evaluator/skydiscover_runpod_eval.py \
  --seed-id mixed_quant_sliding_window \
  --seed-id ema_gptq_lite_qat \
  --seed-id partial_rope_xsa_ema \
  --seed-id sliding_window_eval_only \
  --source-override mixed_quant_sliding_window=experiments/exp01_mixed_export/train_gpt.py \
  --source-override ema_gptq_lite_qat=experiments/exp01_mixed_export/train_gpt.py \
  --source-override partial_rope_xsa_ema=train_gpt.py \
  --source-override sliding_window_eval_only=train_gpt.py
