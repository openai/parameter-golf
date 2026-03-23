#!/bin/bash
set -euo pipefail

# TTT Calibration Sweep — 11 configs in ~45 min
# Goal: hold the 1.11 BPB seen at chunk 51 across the full val set
#
# Strategy: the model hits 1.1107 at chunk 51 then degrades. We sweep:
#   - max_train_chunks: when to stop TTT (before distribution shift)
#   - ema_decay: how much to smooth (0.995 washes everything out)
#   - lr: adaptation speed (lower = less overshoot)
#   - epochs: per-chunk intensity
#   - freeze_blocks: protect layers from distribution shift
#   - momentum: SGD momentum accumulation
#
# Each run: ~3.5-4 min (load int6 checkpoint + TTT eval only)
# Dropped: E(noema_60), J(2ep_50), H(veryslow_40), N(lightema_50) — least likely to win

cd /workspace/parameter-golf
export PYTHONPATH="/workspace/parameter-golf/flash-attention/hopper:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT_PATH:-final_model.int6.ptz}"
RUNNER="ttt_eval_runner.py"
LOGDIR="logs/ttt_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  TTT Calibration Sweep — 11 configs"
echo "  Checkpoint: $CHECKPOINT"
echo "  Logs: $LOGDIR"
echo "============================================"

run_config() {
    local TAG="$1"
    local MAX_CHUNKS="$2"
    local EMA="$3"
    local LR="$4"
    local EPOCHS="$5"
    local FREEZE="$6"
    local MOMENTUM="${7:-0.9}"
    local GRAD_CLIP="${8:-1.0}"

    echo ""
    echo "--- [$TAG] max=$MAX_CHUNKS ema=$EMA lr=$LR ep=$EPOCHS freeze=$FREEZE mom=$MOMENTUM clip=$GRAD_CLIP ---"

    EVAL_ONLY=1 \
    CHECKPOINT_PATH="$CHECKPOINT" \
    TTT_MAX_TRAIN_CHUNKS="$MAX_CHUNKS" \
    TTT_EMA_DECAY="$EMA" \
    TTT_LR="$LR" \
    TTT_EPOCHS="$EPOCHS" \
    TTT_FREEZE_BLOCKS="$FREEZE" \
    TTT_MOMENTUM="$MOMENTUM" \
    TTT_GRAD_CLIP="$GRAD_CLIP" \
    TTT_FREEZE_EMBED=1 \
    SEED=1337 \
    torchrun --standalone --nproc_per_node=8 \
        "$RUNNER" \
        2>&1 | tee "$LOGDIR/${TAG}.log"

    # Extract result
    BPB=$(grep -oP 'legal_ttt_exact val_loss:\S+ val_bpb:\K\S+' "$LOGDIR/${TAG}.log" 2>/dev/null | tail -1)
    echo ">>> $TAG: val_bpb=${BPB:-FAILED}"
    echo "$TAG,$MAX_CHUNKS,$EMA,$LR,$EPOCHS,$FREEZE,$MOMENTUM,$GRAD_CLIP,${BPB:-FAILED}" >> "$LOGDIR/results.csv"
}

# Header
echo "tag,max_chunks,ema_decay,lr,epochs,freeze_blocks,momentum,grad_clip,val_bpb" > "$LOGDIR/results.csv"

# ── BASELINE ──────────────────────────────────────────────────────────────────
run_config "A_baseline"       200  0.995  0.002  3  2  0.9  1.0

# ── CORE HYPOTHESIS: no EMA, vary stop point ──────────────────────────────────
# Stop before distribution shift eats the gains
run_config "B_noema_30"        30  0      0.002  3  2  0.9  1.0
run_config "C_noema_40"        40  0      0.002  3  2  0.9  1.0
run_config "D_noema_50"        50  0      0.002  3  2  0.9  1.0

# ── LOWER LR: gentler adaptation, less overshoot ─────────────────────────────
run_config "E_slow_40"         40  0      0.001  3  2  0.9  1.0
run_config "F_slow_50"         50  0      0.001  3  2  0.9  1.0

# ── FEWER EPOCHS: less per-chunk adaptation ───────────────────────────────────
run_config "G_1ep_40"          40  0      0.002  1  2  0.9  1.0

# ── MORE FREEZE: protect deeper layers from shift ─────────────────────────────
run_config "H_freeze3_40"      40  0      0.002  3  3  0.9  1.0
run_config "I_freeze4_40"      40  0      0.002  3  4  0.9  1.0

# ── LIGHT EMA: smooth but don't wash out ──────────────────────────────────────
run_config "J_lightema_40"     40  0.9    0.002  3  2  0.9  1.0

# ── WILD CARD: aggressive short burst ─────────────────────────────────────────
run_config "K_burst"           20  0      0.005  5  3  0.0  0.5

# ── SUMMARY ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  RESULTS (sorted by BPB)"
echo "============================================"
sort -t',' -k9 -n "$LOGDIR/results.csv" | column -t -s','
echo ""
echo "Full results: $LOGDIR/results.csv"
echo "Logs: $LOGDIR/"
