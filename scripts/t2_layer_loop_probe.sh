#!/usr/bin/env bash
# t2_layer_loop_probe.sh — Pre-flight test for Phase 2.
#
# QUESTION: the training loop at train_gpt_sota_decoded.py flips
#   base_model.looping_active = True
# mid-run once `frac >= h.enable_looping_at` (default 0.35). That changes
# control flow inside the compiled forward. If reduce-overhead mode is active,
# Dynamo will RECOMPILE at that step, which throws away the captured CUDA graph
# and forces a second warmup. If that happens every run, the "graph win" is a
# mirage — we'd be paying double compile tax for a brief graph window.
#
# This script runs a short (400-step) test with
#   TORCH_COMPILE_MODE=reduce-overhead
#   ENABLE_LOOPING_AT=0.25  (guaranteed to fire before the run ends)
#   TORCH_LOGS=recompiles,graph_breaks
# and greps the log for the recompile trigger around the toggle step.
#
# Success criteria:
#   * <= 2 total recompiles (one initial + one post-toggle is ACCEPTABLE)
#   * all recompiles occur in the first 120 steps (warmup + toggle window)
#   * final tok/s >= 95% of peak tok/s (meaning the graph recaptured cleanly)
#
# Run it on Spark only when Tier A / s34 has finished (don't interrupt).

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LABEL="t2_layer_loop_probe"
LOG="logs/sweep/${LABEL}.log"
mkdir -p logs/sweep

bash scripts/run_experiment.sh "$LABEL" \
  QK_GAIN_INIT=5.5 WARMDOWN_FRAC=0.64 \
  TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1 TTT_EPOCHS=1 \
  EMA_DECAY=0.995 LOGIT_SOFTCAP=20 MATRIX_LR=0.042 \
  ITERATIONS=400 MAX_WALLCLOCK_SECONDS=1500 TIMEOUT_SECS=3000 \
  ENABLE_LOOPING_AT=0.25 \
  FAST_SMOKE=0 SEED=42 \
  TORCHDYNAMO_DISABLE=0 TORCH_COMPILE_DISABLE=0 \
  TORCH_COMPILE_MODE=reduce-overhead \
  TORCH_LOGS=recompiles,graph_breaks \
  NOTES=t2_layer_loop_toggle_recompile_probe

echo
echo "-- T2 verdict --"
if [[ ! -f "$LOG" ]]; then
  echo "FAIL: log file not produced"
  exit 2
fi

recompiles=$(grep -c "Recompiling function" "$LOG" || true)
graph_breaks=$(grep -c "Graph break" "$LOG" || true)
toggle_step=$(grep -oE "layer_loop:enabled step:[0-9]+" "$LOG" | head -n1 | grep -oE "[0-9]+" || echo "-")
final_toks=$(grep -oE "tok/s: [0-9.]+" "$LOG" | tail -n1 | grep -oE "[0-9.]+" || echo "-")
peak_toks=$(grep -oE "tok/s: [0-9.]+" "$LOG" | grep -oE "[0-9.]+" | sort -n | tail -n1 || echo "-")

echo "toggle_step=$toggle_step"
echo "recompiles=$recompiles"
echo "graph_breaks=$graph_breaks"
echo "final_tok/s=$final_toks"
echo "peak_tok/s=$peak_toks"

verdict="UNKNOWN"
if [[ "$recompiles" -le 2 ]]; then
  verdict="PASS_recompiles_bounded"
else
  verdict="FAIL_excess_recompiles"
fi
echo "verdict: $verdict"

python3 scripts/graph_probe.py "$LOG"
