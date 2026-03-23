#!/bin/bash
# Run ALL experiments as smoke tests on 1 GPU, output ranked results.
# Usage: CUDA_VISIBLE_DEVICES=2 ./run_priority_smoke.sh [wallclock_seconds]
#
# Default 120s per experiment. ~30 experiments × 2min = ~60 min total.
# Results saved to smoke_results.tsv and printed sorted at the end.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WALLCLOCK="${1:-120}"

# Force single GPU
export NUM_GPUS=1

# All experiments in priority order
EXPERIMENTS=(
    # Tier 1: Safest bets
    "baseline"
    "idea01_byte_weighted_loss"
    "idea12_zloss"
    "idea13_label_smooth"
    "idea16_embed_scales"
    "idea11_swiglu"
    # Tier 2: Medium confidence
    "idea14_layerwise_lr"
    "idea20_ema"
    "idea21_stochastic_depth"
    "idea07_bigram16k"
    "idea19_grad_central"
    "idea22_wsd_schedule"
    "idea25_sandwich_norm"
    "idea24_deepnorm_init"
    "idea26_agc"
    "idea23_batch_warmup"
    # Tier 3: Higher risk, higher potential
    "idea02_factorized_bigram"
    "idea03_entropy_reg"
    "idea04_conditional_resid"
    "idea27_multi_token_pred"
    "idea28_diff_attention"
    "idea09_trigram"
    "idea05_embed_factorize"
    "idea06_adaptive_ns"
    # Tier 4: Quantization / compression
    "idea29_asymmetric_quant"
    "idea30_groupwise_quant"
    "idea18_lzma"
    "idea08_11th_layer"
    "idea17_gqa2kv"
    # Eval-only (no retraining needed, but script still runs full pipeline)
    "idea15_eval_stride32"
)

TOTAL=${#EXPERIMENTS[@]}
RESULTS_FILE="$SCRIPT_DIR/smoke_results.tsv"
echo -e "rank\texperiment\tval_bpb\tartifact_bytes\tstatus" > "$RESULTS_FILE"

echo "========================================"
echo "  PARAMETER GOLF SMOKE TESTS"
echo "  $TOTAL experiments × ${WALLCLOCK}s each"
echo "  Estimated total: ~$((TOTAL * WALLCLOCK / 60)) minutes"
echo "  GPU: ${CUDA_VISIBLE_DEVICES:-all}"
echo "========================================"

DONE=0
FAILED=0
START_ALL=$(date +%s)

for exp in "${EXPERIMENTS[@]}"; do
    DONE=$((DONE + 1))

    if [ ! -f "$SCRIPT_DIR/$exp/train_gpt.py" ]; then
        echo "[$DONE/$TOTAL] SKIP: $exp (no train_gpt.py)"
        echo -e "-\t$exp\t999\t-\tskipped" >> "$RESULTS_FILE"
        continue
    fi

    echo ""
    echo "[$DONE/$TOTAL] ========== $exp =========="

    # Clean previous run
    rm -f "$SCRIPT_DIR/$exp/train_seed42.log" "$SCRIPT_DIR/$exp/final_model"* "$SCRIPT_DIR/$exp/logs"/*

    "$SCRIPT_DIR/run_experiment.sh" "$exp" 42 "$WALLCLOCK" 2>&1 || true

    # Parse results
    LOG="$SCRIPT_DIR/$exp/train_seed42.log"
    if [ -f "$LOG" ]; then
        BPB=$(grep "final_int8_zlib_roundtrip_exact" "$LOG" | grep -oP 'val_bpb:\K[0-9.]+' | tail -1 || echo "")
        BYTES=$(grep "Serialized model int6" "$LOG" | grep -oP '[0-9]+(?= bytes)' | head -1 || echo "")
        if [ -n "$BPB" ]; then
            echo -e "-\t$exp\t$BPB\t${BYTES:-?}\tdone" >> "$RESULTS_FILE"
            echo "  >>> val_bpb=$BPB  artifact=${BYTES:-?} bytes"
        else
            # Maybe crashed before final eval — grab last val_bpb from training
            LAST_BPB=$(grep "val_bpb:" "$LOG" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "")
            if [ -n "$LAST_BPB" ]; then
                echo -e "-\t$exp\t${LAST_BPB}\t-\tpartial" >> "$RESULTS_FILE"
                echo "  >>> PARTIAL (last val_bpb=$LAST_BPB, no quant roundtrip)"
            else
                echo -e "-\t$exp\t999\t-\tcrashed" >> "$RESULTS_FILE"
                echo "  >>> CRASHED (check $LOG)"
                FAILED=$((FAILED + 1))
            fi
        fi
    else
        echo -e "-\t$exp\t999\t-\tno_log" >> "$RESULTS_FILE"
        echo "  >>> NO LOG"
        FAILED=$((FAILED + 1))
    fi
done

END_ALL=$(date +%s)
ELAPSED=$(( (END_ALL - START_ALL) / 60 ))

echo ""
echo ""
echo "================================================================"
echo "  FINAL RESULTS — sorted by val_bpb (best first)"
echo "  $DONE experiments, $FAILED failed, ${ELAPSED} minutes total"
echo "================================================================"
echo ""

# Sort by BPB (column 3), add rank, print as table
{
    echo -e "rank\texperiment\tval_bpb\tartifact\tstatus"
    tail -n +2 "$RESULTS_FILE" | sort -t$'\t' -k3 -n | nl -w2 -s$'\t'
} | column -t -s$'\t'

echo ""
echo "================================================================"
echo "  TOP 5 — candidates for full 3-seed eval"
echo "================================================================"
tail -n +2 "$RESULTS_FILE" | sort -t$'\t' -k3 -n | head -5 | while IFS=$'\t' read -r _ name bpb bytes status; do
    echo "  $name  val_bpb=$bpb  ($status)"
done

echo ""
echo "Next steps:"
echo "  1. Pick top ideas from above"
echo "  2. CUDA_VISIBLE_DEVICES=X ./run_full_3seed.sh <idea_name>"
echo "  3. ./prepare_submission.sh <idea_name> <title> <author> <github>"
echo ""
echo "Results saved: $RESULTS_FILE"
