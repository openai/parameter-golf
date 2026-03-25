#!/bin/bash
set -euo pipefail
# Stage 2 v2: Parallel 1-GPU screens on the SOTA stack.
# 7 slots on 7 GPUs, 150s each, done in ~3 min.
#
# Pivot: all hypotheses are built on the SOTA stack (MLP 3x, int6, sliding eval).
# Control R0 is exact SOTA replay. H1-H6 are single/combined mutations.

PYTHON=/data/pgolf_venv/bin/python
TRAIN=/data/parameter-golf/train_gpt.py
DATA=/data/parameter-golf/data/datasets/fineweb10B_sp1024
TOK=/data/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
LOGDIR=/data/pgolf_runs/s2/screen_v2
mkdir -p "$LOGDIR"

WALLCLOCK=150

# SOTA stack base (from WarmdownQuantization record)
SOTA_ENV=(
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
    TRAIN_SEQ_LEN=2048
    TRAIN_BATCH_TOKENS=786432
    TIED_EMBED_LR=0.03
    MATRIX_LR=0.02
    SCALAR_LR=0.02
    MLP_HIDDEN=1536
    INTX_BITS=6
    INT8_KEEP_TOK_EMB_FP16=1
    INT8_KEEP_LAST_KV_LAYERS_FP16=2
    EVAL_SEQ_LEN=2048
    EVAL_STRIDE=256
    EVAL_BATCH_SEQS=256
)

run_slot() {
    local gpu=$1 slot=$2
    shift 2
    local logfile="$LOGDIR/${slot}.log"
    echo "GPU $gpu: $slot → $logfile"
    (
        for kv in "${SOTA_ENV[@]}"; do export "$kv"; done
        for kv in "$@"; do export "$kv"; done
        export CUDA_VISIBLE_DEVICES=$gpu
        export RUN_ID="s2v2_${slot}"
        $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 "$TRAIN" \
            > "$logfile" 2>&1
    ) &
}

echo "================================================================"
echo " STAGE 2 v2 — SOTA STACK SCREENS — 7 slots, ${WALLCLOCK}s each"
echo "================================================================"
echo ""

# GPU 0: R0 — SOTA stack control (exact replay)
run_slot 0 R0

# GPU 1: H5 — Smaller batch → more optimizer steps
run_slot 1 H5_small_batch TRAIN_BATCH_TOKENS=524288

# GPU 2: H1 — Higher Muon momentum (from our trunk winner)
run_slot 2 H1_high_momentum \
    MUON_MOMENTUM=0.99 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500

# GPU 3: H2 — Longer warmdown (from our trunk winner)
run_slot 3 H2_long_warmdown WARMDOWN_ITERS=3000

# GPU 4: H4 — Muon weight decay
run_slot 4 H4_muon_wd MUON_WEIGHT_DECAY=0.02

# GPU 5: H6 — Denser sliding eval (stride 128 vs 256)
run_slot 5 H6_dense_eval EVAL_STRIDE=128

# GPU 6: H_combined — batch + momentum + warmdown
run_slot 6 H_combined \
    TRAIN_BATCH_TOKENS=524288 \
    MUON_MOMENTUM=0.99 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 \
    WARMDOWN_ITERS=3000

echo ""
echo "All 7 slots launched. Waiting..."
wait
echo ""

echo "================================================================"
echo " SCREEN v2 RESULTS — SOTA stack mutations (relative to R0)"
echo "================================================================"
printf "%-16s %-10s %-8s %-14s %-14s %s\n" "Slot" "step_avg" "steps" "pre_quant" "post_quant" "size_bytes"
echo "------------------------------------------------------------------------"

for slot in R0 H5_small_batch H1_high_momentum H2_long_warmdown H4_muon_wd H6_dense_eval H_combined; do
    log="$LOGDIR/${slot}.log"
    if [ ! -f "$log" ]; then
        printf "%-16s MISSING\n" "$slot"
        continue
    fi
    sa=$(grep "step_avg" "$log" | tail -1 | grep -oP 'step_avg:\K[0-9.]+' 2>/dev/null || echo "N/A")
    sr=$(grep -oP 'step:\K[0-9]+' "$log" | tail -1 2>/dev/null || echo "N/A")
    pre=$(grep "val_bpb" "$log" | grep -v "final_" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "N/A")
    post=$(grep "final_int8_zlib_roundtrip_exact" "$log" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "N/A")
    sz=$(grep "Total submission size int8" "$log" | grep -oP ': \K[0-9]+' 2>/dev/null || echo "N/A")
    printf "%-16s %-10s %-8s %-14s %-14s %s\n" "$slot" "$sa" "$sr" "$pre" "$post" "$sz"
done
