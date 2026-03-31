#!/bin/bash
set -euo pipefail
# TORNADO GRID: Sweep CADENCE, KL_WEIGHT, TEMP around the base concept
# Each arm runs MAX_WALLCLOCK_SECONDS on 8 GPUs, sequentially.
# After all arms, prints a BPB comparison table.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# Use miniconda Python/torchrun (system torchrun is CPU-only)
export PATH="/home/frosty40/miniconda3/bin:${PATH}"

WALLCLOCK="${WALLCLOCK:-200}"          # seconds per arm (default 200s quick test)
NPROC="${NPROC:-8}"
SEED="${SEED:-1337}"
NGRAM_EVAL_SECS="${NGRAM_EVAL_SECS:-90}"  # cap eval time so grid doesn't stall

LOG_DIR="${REPO_ROOT}/logs/tornado_grid_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "  TORNADO GRID SEARCH"
echo "  Wallclock per arm: ${WALLCLOCK}s  |  GPUs: ${NPROC}"
echo "  Logs: ${LOG_DIR}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Grid definition: ARM_ID  CADENCE  KL_WEIGHT  TEMP  LABEL
# ---------------------------------------------------------------------------
#   arm0  = baseline (tornado disabled)
#   arms 1-3 = cadence spine  (KL=0.10, TEMP=2.0)
#   arms 4-5 = KL spine       (CADENCE=4, TEMP=2.0)
#   arms 6-7 = temp spine     (CADENCE=4, KL=0.10)
#   arm8     = aggressive     (CADENCE=2, KL=0.20, TEMP=2.0)
#   arm9     = conservative   (CADENCE=8, KL=0.05, TEMP=4.0)
# ---------------------------------------------------------------------------

declare -a ARM_IDS=(0 1 2 3 4 5 6 7 8 9)
declare -a CADENCES=(0   2   4   8   4    4    4    4    2    8 )
declare -a KL_WTS=(  0   0.10 0.10 0.10 0.05 0.20 0.10 0.10 0.20 0.05)
declare -a TEMPS=(   2.0 2.0  2.0  2.0  2.0  2.0  1.0  4.0  2.0  4.0 )
declare -a LABELS=(
    "baseline__no_tornado"
    "cadence2__kl0.10__t2.0"
    "cadence4__kl0.10__t2.0"
    "cadence8__kl0.10__t2.0"
    "cadence4__kl0.05__t2.0"
    "cadence4__kl0.20__t2.0"
    "cadence4__kl0.10__t1.0"
    "cadence4__kl0.10__t4.0"
    "cadence2__kl0.20__t2.0"
    "cadence8__kl0.05__t4.0"
)

N_ARMS=${#ARM_IDS[@]}
declare -a LOG_FILES=()

for i in "${!ARM_IDS[@]}"; do
    arm="${ARM_IDS[$i]}"
    cadence="${CADENCES[$i]}"
    kl="${KL_WTS[$i]}"
    temp="${TEMPS[$i]}"
    label="${LABELS[$i]}"
    logfile="${LOG_DIR}/arm${arm}_${label}.log"
    LOG_FILES+=("${logfile}")

    echo ""
    echo "--- ARM ${arm}/${N_ARMS} : ${label} ---"
    echo "    CADENCE=${cadence}  KL_WEIGHT=${kl}  TEMP=${temp}"

    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
    COMPLEMENT_ALPHA=0 \
    XSA_LAST_N=11 \
    BIGRAM_VOCAB_SIZE=2048 \
    ROPE_DIMS=16 \
    SWA_EVERY=50 \
    MTP_NUM_HEADS=0 \
    TRIGRAM=1 \
    LATE_QAT_THRESHOLD=0 \
    NGRAM_EVAL_ORDER=9 \
    NGRAM_EVAL_MIN_ORDER=2 \
    NGRAM_EVAL_ADAPTIVE=1 \
    NGRAM_EVAL_ALPHA=0.30 \
    NGRAM_EVAL_ALPHA_MIN=0.05 \
    NGRAM_EVAL_ALPHA_MAX=0.60 \
    NGRAM_EVAL_ENTROPY_CENTER=3.0 \
    NGRAM_EVAL_ENTROPY_SCALE=2.0 \
    NGRAM_EVAL_MIN_COUNT=2 \
    NGRAM_EVAL_BUCKETS=8388608 \
    NGRAM_EVAL_MAX_SECONDS="${NGRAM_EVAL_SECS}" \
    NGRAM_ENTROPY_SHIFT=1 \
    NGRAM_ORDER_MULTS="0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0" \
    CUBRIC_CADENCE=0 \
    TORNADO_CADENCE="${cadence}" \
    TORNADO_KL_WEIGHT="${kl}" \
    TORNADO_TEMP="${temp}" \
    torchrun --standalone --nproc_per_node="${NPROC}" \
        "${SCRIPT_DIR}/train_gpt.py" \
        2>&1 | tee "${logfile}"

    echo "    done -> ${logfile}"
done

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  TORNADO GRID RESULTS  (seed=${SEED}  wallclock=${WALLCLOCK}s)"
echo "========================================================"
printf "%-4s  %-32s  %-8s  %-8s  %-10s  %-10s  %s\n" \
    "ARM" "LABEL" "CADENCE" "KL" "BASE_BPB" "NGRAM_BPB" "DELTA"
echo "------------------------------------------------------------------------"

baseline_ngram_bpb=""
baseline_base_bpb=""

for i in "${!ARM_IDS[@]}"; do
    arm="${ARM_IDS[$i]}"
    cadence="${CADENCES[$i]}"
    kl="${KL_WTS[$i]}"
    label="${LABELS[$i]}"
    logfile="${LOG_FILES[$i]}"

    # Base BPB (no n-gram): final_sliding_window_exact
    base_bpb=$(grep -oP 'final_sliding_window_exact val_bpb:\K[\d.]+' "${logfile}" 2>/dev/null | tail -1 || echo "N/A")

    # N-gram BPB: prefer _exact, fall back to _partial
    ngram_bpb=$(grep -oP "final_sliding_window_ngram9_exact val_bpb:\K[\d.]+" "${logfile}" 2>/dev/null | tail -1 \
             || grep -oP "final_sliding_window_ngram9_partial val_bpb:\K[\d.]+" "${logfile}" 2>/dev/null | tail -1 \
             || echo "N/A")

    # Compute delta vs baseline
    if [ "${arm}" -eq 0 ]; then
        baseline_base_bpb="${base_bpb}"
        baseline_ngram_bpb="${ngram_bpb}"
        delta="(baseline)"
    else
        if [ "${ngram_bpb}" != "N/A" ] && [ "${baseline_ngram_bpb}" != "N/A" ] && [ "${baseline_ngram_bpb}" != "" ]; then
            delta=$(python3 -c "print(f'{float(\"${ngram_bpb}\") - float(\"${baseline_ngram_bpb}\"):+.4f}')" 2>/dev/null || echo "N/A")
        else
            delta="N/A"
        fi
    fi

    printf "%-4s  %-32s  %-8s  %-8s  %-10s  %-10s  %s\n" \
        "${arm}" "${label:0:32}" "${cadence}" "${kl}" "${base_bpb}" "${ngram_bpb}" "${delta}"
done

echo "========================================================"
echo "  negative delta = improvement over baseline"
echo "  Logs saved to: ${LOG_DIR}"
echo "========================================================"
