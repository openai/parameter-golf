#!/bin/bash
set -uo pipefail

PYTHON="/home/nesta/parameter-golf/.venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

set -a; source /home/nesta/parameter-golf/.env; set +a

RESULTS_FILE="/home/nesta/parameter-golf/grid_search_results.csv"
echo "passes,jacobian,bpb_0,bpb_25,bpb_50,int6_bpb,step_avg_ms,mem_mib,growth_pass2_step50" > "$RESULTS_FILE"

export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export DATA_PATH="../../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../../data/tokenizers/fineweb_1024_bpe.model"
export SEED=1337
export ITERATIONS=50
export MAX_WALLCLOCK_SECONDS=900
export VAL_LOSS_EVERY=25
export TRAIN_LOG_EVERY=10
export WARMUP_STEPS=5
export WARMDOWN_ITERS=10
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export EVAL_STRIDE=0
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export BIGRAM_VOCAB_SIZE=1536
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=5
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3
export SWA_ENABLED=0
export LATE_QAT=0
export TTT_ENABLED=0
export CORE_START=3
export CORE_END=8
export CORE_QUANT_ENABLED=0
export WANDB_PROJECT="parameter-golf"

RUN_IDX=0
for PASSES in 2 3 4; do
    for JAC in 0.0 0.001 0.01 0.1; do
        RUN_IDX=$((RUN_IDX + 1))
        export NUM_PASSES=$PASSES
        export WANDB_NAME="grid_p${PASSES}_j${JAC}"
        LOG="/home/nesta/parameter-golf/grid_p${PASSES}_j${JAC}.log"

        if [ -f "$LOG" ] && grep -q "final_int6_roundtrip_exact" "$LOG"; then
            echo "[$RUN_IDX/12] SKIP (already done): passes=$PASSES jac=$JAC"
        else
            echo "[$RUN_IDX/12] START passes=$PASSES jac=$JAC  ($(date +%H:%M:%S))"
            $PYTHON train_gpt_recurrent.py \
                --feedback-mode diagonal --feedback-rank 2 \
                --residual-scale-init 0.5 \
                --jacobian-proxy-weight "$JAC" \
                > "$LOG" 2>&1 || {
                    echo "[$RUN_IDX/12] FAILED passes=$PASSES jac=$JAC (exit=$?)"
                    echo "$PASSES,$JAC,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$RESULTS_FILE"
                    continue
                }
            echo "[$RUN_IDX/12] DONE passes=$PASSES jac=$JAC ($(date +%H:%M:%S))"
        fi

        BPB_0=$(grep 'step:0/.*val_bpb:' "$LOG" | head -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/' || echo "N/A")
        BPB_25=$(grep 'step:25/.*val_bpb:' "$LOG" | head -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/' || echo "N/A")
        BPB_50=$(grep 'step:50/.*val_bpb:' "$LOG" | head -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/' || echo "N/A")
        INT6_BPB=$(grep 'final_int6_roundtrip_exact.*val_bpb:' "$LOG" | head -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/' || echo "N/A")
        STEP_AVG=$(grep 'step:50/.*step_avg:' "$LOG" | head -1 | sed 's/.*step_avg:\([0-9.]*\)ms.*/\1/' || echo "N/A")
        MEM=$(grep 'peak memory' "$LOG" | head -1 | sed 's/.*allocated: \([0-9]*\) MiB.*/\1/' || echo "N/A")
        GROWTH_50=$(grep 'step:50/.*growth=' "$LOG" | head -1 | sed "s/.*growth=\[//;s/\].*//;s/'//g" || echo "N/A")

        echo "$PASSES,$JAC,$BPB_0,$BPB_25,$BPB_50,$INT6_BPB,$STEP_AVG,$MEM,$GROWTH_50" >> "$RESULTS_FILE"
        echo "  => bpb@50=$BPB_50 int6_bpb=$INT6_BPB step_avg=${STEP_AVG}ms mem=${MEM}MiB"
    done
done

echo ""
echo "=== ALL 12 RUNS COMPLETE ($(date)) ==="
echo "Results CSV: $RESULTS_FILE"
cat "$RESULTS_FILE"
