#!/usr/bin/env bash
# graph_offensive_launcher.sh — queued launcher for the CUDA-graph offensive.
#
# Waits until no train_gpt process is running on THIS host (Spark), then runs:
#   1. T2 layer-loop probe (400 steps) — gates whether reduce-overhead is even viable
#   2. The full graph_offensive.tsv (8 rows × ~30min each, ~4h total) IFF T2 passes
#   3. graph_probe.py summary across all logs
#
# Safe to arm while other machines (AGX, Nano) are still training — this only
# polls the local machine for train_gpt_* processes. It will NOT
# interrupt them.
#
# Armed via:
#   nohup bash scripts/graph_offensive_launcher.sh > logs/sweep/graph_offensive_launcher.log 2>&1 &
#   disown

set -u

# Single-instance lock (prevent duplicate launches).
LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd "$(dirname "$0")/.."
LOG=logs/sweep/graph_offensive_launcher.log
SWEEP=scripts/sweeps/graph_offensive.tsv
mkdir -p logs/sweep

echo "[$(date)] graph-offensive launcher armed (pid=$$)" >> "$LOG"

# Wait for any running training on this host to exit.
while pgrep -f 'train_gpt_sota_decoded\.py|train_gpt_agx\.py|train_gpt_mlx\.py|train_gpt\.py' >/dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] Spark free; running T2 layer-loop probe" >> "$LOG"

# --- Step 1: T2 probe ---
bash scripts/t2_layer_loop_probe.sh >> "$LOG" 2>&1
T2_LOG=logs/sweep/t2_layer_loop_probe.log
T2_RECOMPS=$(grep -c "Recompiling function" "$T2_LOG" 2>/dev/null || echo 999)
echo "[$(date)] T2 recompiles=${T2_RECOMPS}" >> "$LOG"

if [[ "$T2_RECOMPS" -gt 4 ]]; then
    echo "[$(date)] T2 FAILED (recompiles>${T2_RECOMPS}) — SKIPPING full sweep, leaving control runs only" >> "$LOG"
    # Still run the two default-mode rows so we at least document eager-vs-default delta
    bash scripts/run_experiment.sh graph_off_00_eager \
        QK_GAIN_INIT=5.5 WARMDOWN_FRAC=0.64 TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1 TTT_EPOCHS=1 \
        EMA_DECAY=0.995 LOGIT_SOFTCAP=20 MATRIX_LR=0.042 ITERATIONS=500 \
        MAX_WALLCLOCK_SECONDS=1800 TIMEOUT_SECS=3600 FAST_SMOKE=0 SEED=42 \
        TORCH_LOGS=recompiles,graph_breaks NOTES=eager_compile_disabled >> "$LOG" 2>&1
    bash scripts/run_experiment.sh graph_off_01_default_ctrl \
        QK_GAIN_INIT=5.5 WARMDOWN_FRAC=0.64 TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1 TTT_EPOCHS=1 \
        EMA_DECAY=0.995 LOGIT_SOFTCAP=20 MATRIX_LR=0.042 ITERATIONS=500 \
        MAX_WALLCLOCK_SECONDS=1800 TIMEOUT_SECS=3600 FAST_SMOKE=0 SEED=42 \
        TORCHDYNAMO_DISABLE=0 TORCH_COMPILE_DISABLE=0 \
        TORCH_LOGS=recompiles,graph_breaks NOTES=compile_default_baseline >> "$LOG" 2>&1
else
    echo "[$(date)] T2 passed — launching full graph_offensive sweep" >> "$LOG"
    bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1
fi

# --- Step 2: summary ---
echo "[$(date)] all runs complete; building summary" >> "$LOG"
python3 scripts/graph_probe.py logs/sweep/graph_off_*.log >> "$LOG" 2>&1 || true

echo "[$(date)] graph-offensive launcher DONE" >> "$LOG"
