#!/bin/bash
set -euo pipefail
# Stage 2 phased launcher for 8xH100.
#
# Three horizons:
#   sanity   → 90s,  catch crashes / bad exports
#   screen   → 150s, rank ideas cheaply (ms/step, steps, early BPB trend)
#   decision → 600s, full leaderboard-comparable run
#
# Usage:
#   bash launch_s2.sh sanity   S2-B0          # single slot sanity
#   bash launch_s2.sh screen   S2-E1          # single slot screen
#   bash launch_s2.sh decision S2-B0          # full 10 min
#   bash launch_s2.sh gate                    # S2-B0 sanity then full
#   bash launch_s2.sh screen-all              # screen E1-E6 sequentially
#   bash launch_s2.sh promote  S2-E2 S2-E3   # full 10 min on survivors

PYTHON=/data/pgolf_venv/bin/python
TRAIN=/data/parameter-golf/train_gpt.py
DATA=/data/parameter-golf/data/datasets/fineweb10B_sp1024
TOK=/data/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
LOGDIR=/data/pgolf_runs/s2
mkdir -p "$LOGDIR"

# ── Shared defaults ──
export TRAIN_LOG_EVERY=50
export VAL_LOSS_EVERY=200
export VOCAB_SIZE=1024
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MODEL_DIM=512
export NUM_LAYERS=9
export TIE_EMBEDDINGS=1
export DATA_PATH="$DATA"
export TOKENIZER_PATH="$TOK"

# ── Wallclock by horizon ──
SANITY_SECS=90
SCREEN_SECS=150
DECISION_SECS=600

# ── Slot env setup ──
# Each function sets slot-specific env vars on top of defaults.
# The trunk (S2-B0) env is the base for all E-slots.

set_trunk_env() {
    export TRAIN_SEQ_LEN=4096
    export TRAIN_BATCH_TOKENS=393216
    export TIED_EMBED_LR=0.03
    export MATRIX_LR=0.02
    export SCALAR_LR=0.02
    export MUON_MOMENTUM=0.99
    export MUON_MOMENTUM_WARMUP_START=0.92
    export MUON_MOMENTUM_WARMUP_STEPS=1500
    export WARMDOWN_ITERS=3000
}

set_slot_env() {
    local slot=$1

    # Always start from clean trunk
    set_trunk_env

    case "$slot" in
        S2-B0)
            # Pure trunk replay. No overrides.
            ;;
        S2-E1)
            # H1: sliding eval helps on the clean trunk
            export EVAL_STRIDE=64
            export EVAL_BATCH_SEQS=128
            ;;
        S2-E2)
            # H2: fp16 tied-embedding export helps on the clean trunk
            export INT8_KEEP_TOK_EMB_FP16=1
            export MLP_HIDDEN=992
            ;;
        S2-E3)
            # H3: our quant-quality mutation transfers to 8xH100
            export INT8_CLIP_PERCENTILE=99.99995
            export INT8_PER_ROW_SCALE_DTYPE=float32
            ;;
        S2-E4)
            # H4: Muon WD helps on top of fp16-embed trunk
            export INT8_KEEP_TOK_EMB_FP16=1
            export MLP_HIDDEN=992
            export MUON_WEIGHT_DECAY=0.02
            ;;
        S2-E6)
            # H5: adaptive Muon lowers ms/step enough to buy more steps
            export INT8_KEEP_TOK_EMB_FP16=1
            export MLP_HIDDEN=992
            export ADAPTIVE_MUON_BACKEND=1
            export MUON_BACKEND_STEPS=5
            export MUON_BACKEND_STEPS_EARLY=3
            export MUON_BACKEND_WARMUP_STEPS=2000
            ;;
        S2-E6B)
            # Record-derivative: 2048 + int6 + adaptive Muon
            export TRAIN_SEQ_LEN=2048
            export TRAIN_BATCH_TOKENS=786432
            export TIED_EMBED_LR=0.03
            export MATRIX_LR=0.02
            export SCALAR_LR=0.02
            export MUON_BACKEND_STEPS=5
            export MLP_HIDDEN=1536
            export INTX_BITS=6
            export INT8_KEEP_TOK_EMB_FP16=1
            export INT8_KEEP_LAST_KV_LAYERS_FP16=2
            export EVAL_SEQ_LEN=2048
            export EVAL_STRIDE=256
            export EVAL_BATCH_SEQS=256
            export ADAPTIVE_MUON_BACKEND=1
            export MUON_BACKEND_STEPS_EARLY=3
            export MUON_BACKEND_WARMUP_STEPS=2000
            # Clear trunk-specific vars that don't apply
            unset MUON_MOMENTUM MUON_MOMENTUM_WARMUP_START MUON_MOMENTUM_WARMUP_STEPS WARMDOWN_ITERS
            ;;
        S2-E8)
            # Warmdown revisit: conservative schedule challenge
            export INT8_KEEP_TOK_EMB_FP16=1
            export MLP_HIDDEN=992
            export WARMDOWN_ITERS=20000
            export TIED_EMBED_LR=0.05
            export MATRIX_LR=0.04
            export SCALAR_LR=0.04
            export GRAD_CLIP_NORM=1.0
            export EVAL_SEQ_LEN=1408
            ;;
        *)
            echo "ERROR: unknown slot $slot"
            exit 1
            ;;
    esac
}

# ── Run a single slot at a given horizon ──
run_one() {
    local slot=$1
    local horizon=$2

    case "$horizon" in
        sanity)   export MAX_WALLCLOCK_SECONDS=$SANITY_SECS ;;
        screen)   export MAX_WALLCLOCK_SECONDS=$SCREEN_SECS ;;
        decision) export MAX_WALLCLOCK_SECONDS=$DECISION_SECS ;;
        *)        echo "ERROR: unknown horizon $horizon"; exit 1 ;;
    esac

    set_slot_env "$slot"
    export RUN_ID="s2_${slot}_${horizon}"

    local logfile="$LOGDIR/${slot}_${horizon}.log"
    echo ""
    echo "================================================================"
    echo " $slot | $horizon | wallclock=${MAX_WALLCLOCK_SECONDS}s"
    echo " log: $logfile"
    echo "================================================================"

    $PYTHON -m torch.distributed.run \
        --standalone --nproc_per_node=8 \
        "$TRAIN" 2>&1 | tee "$logfile"

    echo ""
    echo "── $slot $horizon DONE ──"

    # Extract key metrics from log
    local step_avg steps_reached final_bpb
    step_avg=$(grep "step_avg" "$logfile" | tail -1 | grep -oP 'step_avg:\K[0-9.]+' || echo "N/A")
    steps_reached=$(grep -oP 'step:\K[0-9]+' "$logfile" | tail -1 || echo "N/A")
    final_bpb=$(grep "final_int8_zlib_roundtrip_exact" "$logfile" | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")

    printf "  %-12s step_avg=%-8s steps=%-8s post_quant_bpb=%s\n" \
        "$slot" "$step_avg" "$steps_reached" "$final_bpb"
}

# ── Modes ──
MODE=${1:-help}
shift || true

case "$MODE" in
    sanity|screen|decision)
        # Run one or more named slots at the given horizon
        for slot in "$@"; do
            run_one "$slot" "$MODE"
        done
        ;;

    gate)
        # S2-B0: sanity first, then full decision if sanity passes
        run_one S2-B0 sanity
        echo ""
        echo "Sanity passed. Running S2-B0 full decision (10 min)..."
        run_one S2-B0 decision
        ;;

    screen-all)
        # Screen all single-factor extensions sequentially
        for slot in S2-E1 S2-E2 S2-E3 S2-E4 S2-E6; do
            run_one "$slot" screen
        done
        echo ""
        echo "================================================================"
        echo " SCREEN SUMMARY"
        echo "================================================================"
        printf "%-8s %-10s %-10s %s\n" "Slot" "step_avg" "steps" "post_quant_bpb"
        echo "----------------------------------------------"
        for slot in S2-E1 S2-E2 S2-E3 S2-E4 S2-E6; do
            local_log="$LOGDIR/${slot}_screen.log"
            sa=$(grep "step_avg" "$local_log" | tail -1 | grep -oP 'step_avg:\K[0-9.]+' 2>/dev/null || echo "N/A")
            sr=$(grep -oP 'step:\K[0-9]+' "$local_log" | tail -1 2>/dev/null || echo "N/A")
            fb=$(grep "final_int8_zlib_roundtrip_exact" "$local_log" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "N/A")
            printf "%-8s %-10s %-10s %s\n" "$slot" "$sa" "$sr" "$fb"
        done
        ;;

    promote)
        # Full 10 min on named survivors
        for slot in "$@"; do
            run_one "$slot" decision
        done
        ;;

    help|*)
        echo "Stage 2 Phased Launcher"
        echo ""
        echo "Usage:"
        echo "  bash launch_s2.sh gate                     # S2-B0 sanity→full"
        echo "  bash launch_s2.sh screen-all               # screen E1-E6 (2.5min each)"
        echo "  bash launch_s2.sh sanity    S2-E1 S2-E2    # 90s sanity check"
        echo "  bash launch_s2.sh screen    S2-E1          # 2.5min screen"
        echo "  bash launch_s2.sh decision  S2-B0          # full 10 min"
        echo "  bash launch_s2.sh promote   S2-E2 S2-E3    # full 10 min on survivors"
        echo ""
        echo "Slots: S2-B0 S2-E1 S2-E2 S2-E3 S2-E4 S2-E6 S2-E6B S2-E8"
        echo ""
        echo "Order:"
        echo "  1. gate         → S2-B0 baseline (must reproduce ~1.2014)"
        echo "  2. screen-all   → rank E1-E6 cheaply"
        echo "  3. promote      → full runs on top 2-3 survivors"
        echo "  4. decision     → composite or S2-E6B record-derivative"
        ;;
esac
