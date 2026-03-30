#!/bin/bash
# smoke_test.sh — Rascal_Stripper 4-way A/B
# Runs: baseline → turbomuon → engramlite → combo  (3200 steps each = 12800 total)
# Val BPB logged every 300 steps. Final sliding-window s64 BPB printed at the end.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC=${NPROC:-8}
LOG_DIR="$SCRIPT_DIR/smoke_logs"
mkdir -p "$LOG_DIR"

run_variant() {
    local name=$1
    local script=$2
    local log="$LOG_DIR/${name}.log"

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  SMOKE: $name  (3200 steps)"
    echo "══════════════════════════════════════════════════"

    ITERATIONS=3200 \
    WARMDOWN_ITERS=800 \
    TRAIN_LOG_EVERY=50 \
    VAL_LOSS_EVERY=300 \
    MAX_WALLCLOCK_SECONDS=0 \
    SKIP_FINAL_EVAL=0 \
    EVAL_STRIDE=64 \
    POST_EMA_DIAGNOSTIC=0 \
    RUN_ID="smoke_${name}" \
    torchrun --nproc_per_node="$NPROC" "$script" \
        2>&1 | tee "$log" | grep --line-buffered -E "step:|val_bpb|val_loss|final_sliding"

    echo "  ── $name done ──"
}

run_variant "baseline"   "$SCRIPT_DIR/train_gpt_safe.py"
run_variant "turbomuon"  "$SCRIPT_DIR/train_gpt_turbomuon.py"
run_variant "engramlite" "$SCRIPT_DIR/train_gpt_engramlite.py"
run_variant "combo"      "$SCRIPT_DIR/train_gpt_combo.py"

echo ""
echo "══════════════════════════════════════════════════"
echo "  RESULTS — final_sliding_window_s64 BPB"
echo "══════════════════════════════════════════════════"
for name in baseline turbomuon engramlite combo; do
    log="$LOG_DIR/${name}.log"
    bpb=$(grep "final_sliding_window_exact" "$log" 2>/dev/null \
          | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    printf "  %-12s  %s\n" "$name" "$bpb"
done
echo "══════════════════════════════════════════════════"
