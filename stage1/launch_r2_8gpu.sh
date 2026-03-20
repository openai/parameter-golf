#!/bin/bash
set -euo pipefail
# Launch all 8 R2 frontier family slots on 8 GPUs (one slot per GPU).
# Usage: bash launch_r2_8gpu.sh
# Expects: /data/pgolf_venv, /data/parameter-golf with data downloaded.

PYTHON=/data/pgolf_venv/bin/python
TRAIN=/data/parameter-golf/train_gpt.py
DATA=/data/parameter-golf/data/datasets/fineweb10B_sp1024
TOK=/data/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
LOGDIR=/data/pgolf_runs/r2
mkdir -p "$LOGDIR"

# Shared defaults
export MAX_WALLCLOCK_SECONDS=600
export TRAIN_LOG_EVERY=50
export VAL_LOSS_EVERY=200
export VOCAB_SIZE=1024
export NUM_HEADS=8
export TIE_EMBEDDINGS=1
export DATA_PATH="$DATA"
export TOKENIZER_PATH="$TOK"

run_slot() {
    local gpu=$1 slot=$2 name=$3
    shift 3
    local logfile="$LOGDIR/${slot}_${name}.log"
    echo "=== GPU $gpu: $slot $name ===" | tee "$logfile"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        export RUN_ID="r2_${slot}_${name}"
        # Apply slot-specific env vars (passed as KEY=VAL args)
        for kv in "$@"; do export "$kv"; done
        $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 "$TRAIN" \
            >> "$logfile" 2>&1
        echo "=== GPU $gpu: $slot $name DONE ===" >> "$logfile"
    ) &
}

# P0: baseline (no overrides)
run_slot 0 P0 baseline

# P1: seq2048 training
run_slot 1 P1 m02_seq2048_train \
    TRAIN_SEQ_LEN=2048 TIED_EMBED_LR=0.04 MATRIX_LR=0.032 SCALAR_LR=0.032

# P2: fp16 embed export
run_slot 2 P2 m09_fp16_embed_export \
    INT8_KEEP_TOK_EMB_FP16=1 MLP_HIDDEN=992

# P3: AdamW partition
run_slot 3 P3 m06_adamw_partition \
    TOKEN_OPTIMIZER=adamw SCALAR_OPTIMIZER=adamw TOKEN_WEIGHT_DECAY=0.01 SCALAR_WEIGHT_DECAY=0.01

# P4: sliding eval stride=64
run_slot 4 P4 m11_sliding_eval64 \
    EVAL_STRIDE=64 EVAL_BATCH_SEQS=256

# P5: always-decay (fixed: lower LRs to avoid blowup seen on A100)
run_slot 5 P5 m08_always_decay \
    WARMDOWN_ITERS=20000 MATRIX_LR=0.04 TIED_EMBED_LR=0.05 SCALAR_LR=0.04 GRAD_CLIP_NORM=1.0

# P6: depth10 width480
run_slot 6 P6 m01_depth10_width480 \
    NUM_LAYERS=10 MODEL_DIM=480 NUM_KV_HEADS=4 MLP_MULT=2

# P7: adaptive Muon
run_slot 7 P7 m07_adaptive_muon \
    ADAPTIVE_MUON_BACKEND=1 MUON_BACKEND_STEPS=5 MUON_BACKEND_STEPS_EARLY=3 MUON_BACKEND_WARMUP_STEPS=2000

echo "All 8 slots launched. Logs in $LOGDIR/"
echo "Monitor: tail -f $LOGDIR/*.log"
echo "Wait for completion..."
wait

echo ""
echo "============================================================"
echo "R2 FRONTIER FAMILY — 1xH100 RESULTS (one GPU per slot)"
echo "============================================================"
printf "%-6s %-25s %s\n" "Slot" "Name" "val_bpb"
echo "--------------------------------------------------------------"
for log in "$LOGDIR"/*.log; do
    slot=$(basename "$log" .log | cut -d_ -f1)
    name=$(basename "$log" .log | cut -d_ -f2-)
    bpb=$(grep "final_int8_zlib_roundtrip_exact" "$log" 2>/dev/null | grep -o "val_bpb:[0-9.]*" | cut -d: -f2 || echo "FAILED")
    printf "%-6s %-25s %s\n" "$slot" "$name" "$bpb"
done
