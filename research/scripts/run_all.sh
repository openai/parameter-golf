#!/usr/bin/env bash
# One-shot analysis orchestrator for a spec-006-style run dir.
#
# Expects layout:
#   $RUN_DIR/
#     train.log
#     checkpoints/
#       ckpt_event_step{100,200,...,4500}.pt
#       ckpt_warmdown_start_step*.pt        (skipped by default — uneven window)
#       ckpt_pre_recurrence_step*.pt        (skipped)
#       ckpt_final_{pre,post}_ema_step*.pt  (skipped)
#
# Produces $RUN_DIR/analysis/:
#   train_loss.csv, val_loss.csv, grad_norms.csv, events.txt,
#   delta_matrix.csv, loss_curves.png, grad_norms.png,
#   per_layer_movement.png, lr_normalized_movement.png,
#   loop_differential.png, train_val_gap.png
#
# Usage: ./run_all.sh /path/to/run_dir
set -euo pipefail
RUN_DIR="${1:?usage: run_all.sh <run_dir>}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$RUN_DIR/analysis"
mkdir -p "$OUT"

echo "=== 1/3 parse train.log ==="
python3 "$SCRIPTS_DIR/parse_train_log.py" "$RUN_DIR/train.log" --outdir "$OUT"

echo "=== 2/3 windowed weight delta ==="
python3 "$SCRIPTS_DIR/windowed_weight_delta.py" "$RUN_DIR/checkpoints" --outdir "$OUT"

echo "=== 3/3 plots ==="
python3 "$SCRIPTS_DIR/plots.py" "$OUT"

echo "done → $OUT"
ls -la "$OUT"
