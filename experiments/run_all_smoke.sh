#!/bin/bash
# Run all experiments as 2-minute smoke tests, sequentially
# Usage: ./run_all_smoke.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    "baseline"
    "idea01_byte_weighted_loss"
    "idea02_factorized_bigram"
    "idea03_entropy_reg"
    "idea04_conditional_resid"
    "idea05_embed_factorize"
    "idea06_adaptive_ns"
    "idea07_bigram16k"
    "idea08_11th_layer"
    "idea09_trigram"
)

RESULTS_FILE="$SCRIPT_DIR/smoke_results.tsv"
echo -e "experiment\tseed\tval_bpb\tartifact_bytes\tstatus" > "$RESULTS_FILE"

for exp in "${EXPERIMENTS[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$exp/train_gpt.py" ]; then
        echo "SKIP: $exp (no train_gpt.py)"
        echo -e "$exp\t42\t-\t-\tskipped" >> "$RESULTS_FILE"
        continue
    fi
    echo ""
    echo "========================================"
    echo "  SMOKE TEST: $exp"
    echo "========================================"

    "$SCRIPT_DIR/run_experiment.sh" "$exp" 42 120 || true

    # Parse results
    LOG="$SCRIPT_DIR/$exp/train_seed42.log"
    if [ -f "$LOG" ]; then
        BPB=$(grep "final_int8_zlib_roundtrip_exact" "$LOG" | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
        BYTES=$(grep "Serialized model int6" "$LOG" | grep -oP '[0-9]+ bytes' | head -1 || echo "N/A")
        echo -e "$exp\t42\t$BPB\t$BYTES\tdone" >> "$RESULTS_FILE"
    else
        echo -e "$exp\t42\t-\t-\tfailed" >> "$RESULTS_FILE"
    fi
done

echo ""
echo "========================================"
echo "  SMOKE TEST RESULTS"
echo "========================================"
column -t -s$'\t' "$RESULTS_FILE"
