#!/bin/bash
set -euo pipefail
# Parallel 1-GPU screens: 6 slots on 6 GPUs, 150s each, done in 2.5 min.
# Comparison is relative to S2-B0 running on the same hardware at the same time.

PYTHON=/data/pgolf_venv/bin/python
TRAIN=/data/parameter-golf/train_gpt.py
DATA=/data/parameter-golf/data/datasets/fineweb10B_sp1024
TOK=/data/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
LOGDIR=/data/pgolf_runs/s2/screen
mkdir -p "$LOGDIR"

WALLCLOCK=150

# Shared defaults
BASE_ENV=(
    MAX_WALLCLOCK_SECONDS=$WALLCLOCK
    TRAIN_LOG_EVERY=50
    VAL_LOSS_EVERY=200
    VOCAB_SIZE=1024
    NUM_HEADS=8
    NUM_KV_HEADS=4
    MODEL_DIM=512
    NUM_LAYERS=9
    TIE_EMBEDDINGS=1
    DATA_PATH=$DATA
    TOKENIZER_PATH=$TOK
    TRAIN_SEQ_LEN=4096
    TRAIN_BATCH_TOKENS=393216
    TIED_EMBED_LR=0.03
    MATRIX_LR=0.02
    SCALAR_LR=0.02
    MUON_MOMENTUM=0.99
    MUON_MOMENTUM_WARMUP_START=0.92
    MUON_MOMENTUM_WARMUP_STEPS=1500
    WARMDOWN_ITERS=3000
)

run_slot() {
    local gpu=$1 slot=$2
    shift 2
    local logfile="$LOGDIR/${slot}.log"
    echo "GPU $gpu: $slot → $logfile"
    (
        for kv in "${BASE_ENV[@]}"; do export "$kv"; done
        # Apply slot-specific overrides
        for kv in "$@"; do export "$kv"; done
        export CUDA_VISIBLE_DEVICES=$gpu
        export RUN_ID="s2_screen_${slot}"
        $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 "$TRAIN" \
            > "$logfile" 2>&1
    ) &
}

echo "================================================================"
echo " PARALLEL 1-GPU SCREEN — 6 slots, ${WALLCLOCK}s each"
echo "================================================================"
echo ""

# GPU 0: S2-B0 baseline (no overrides — apples-to-apples reference)
run_slot 0 S2-B0

# GPU 1: S2-E1 sliding eval
run_slot 1 S2-E1 EVAL_STRIDE=64 EVAL_BATCH_SEQS=128

# GPU 2: S2-E2 fp16 embed export
run_slot 2 S2-E2 INT8_KEEP_TOK_EMB_FP16=1 MLP_HIDDEN=992

# GPU 3: S2-E3 quant quality
run_slot 3 S2-E3 INT8_CLIP_PERCENTILE=99.99995 INT8_PER_ROW_SCALE_DTYPE=float32

# GPU 4: S2-E4 Muon WD (on top of fp16 embed)
run_slot 4 S2-E4 INT8_KEEP_TOK_EMB_FP16=1 MLP_HIDDEN=992 MUON_WEIGHT_DECAY=0.02

# GPU 5: S2-E6 adaptive Muon (on top of fp16 embed)
run_slot 5 S2-E6 INT8_KEEP_TOK_EMB_FP16=1 MLP_HIDDEN=992 \
    ADAPTIVE_MUON_BACKEND=1 MUON_BACKEND_STEPS=5 \
    MUON_BACKEND_STEPS_EARLY=3 MUON_BACKEND_WARMUP_STEPS=2000

echo ""
echo "All 6 slots launched. Waiting..."
wait
echo ""

echo "================================================================"
echo " SCREEN RESULTS — relative to S2-B0"
echo "================================================================"
printf "%-8s %-10s %-8s %-14s %-14s %s\n" "Slot" "step_avg" "steps" "pre_quant" "post_quant" "size_bytes"
echo "------------------------------------------------------------------------"

for slot in S2-B0 S2-E1 S2-E2 S2-E3 S2-E4 S2-E6; do
    log="$LOGDIR/${slot}.log"
    if [ ! -f "$log" ]; then
        printf "%-8s MISSING\n" "$slot"
        continue
    fi
    sa=$(grep "step_avg" "$log" | tail -1 | grep -oP 'step_avg:\K[0-9.]+' 2>/dev/null || echo "N/A")
    sr=$(grep -oP 'step:\K[0-9]+' "$log" | tail -1 2>/dev/null || echo "N/A")
    # Last val_bpb before export
    pre=$(grep "val_bpb" "$log" | grep -v "final_" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "N/A")
    # Post-quant
    post=$(grep "final_int8_zlib_roundtrip_exact" "$log" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "N/A")
    sz=$(grep "Total submission size int8" "$log" | grep -oP ': \K[0-9]+' 2>/dev/null || echo "N/A")
    printf "%-8s %-10s %-8s %-14s %-14s %s\n" "$slot" "$sa" "$sr" "$pre" "$post" "$sz"
done
