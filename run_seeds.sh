#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Multi-Seed Runner for DepthRecurrentQAT Statistical Validation
# ============================================================================
#
# Runs train_gpt.py with multiple seeds to measure variance and statistical
# significance of BPB improvement over the NaiveBaseline.
#
# Required environment variables (or pass as arguments):
#   DATA_PATH       - Path to fineweb10B_sp1024 dataset
#   TOKENIZER_PATH  - Path to fineweb_1024_bpe.model tokenizer
#
# Optional:
#   NUM_SEEDS       - Number of seeds to run (default: 5)
#   NPROC           - GPUs per node (default: 8)
#
# Usage:
#   export DATA_PATH=/path/to/fineweb10B_sp1024
#   export TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model
#   ./run_seeds.sh
#
#   # Or pass as arguments:
#   ./run_seeds.sh /path/to/data /path/to/tokenizer [num_seeds]
#
# The script runs 5 seeds sequentially, each taking ~10 minutes (total ~50 min).
# Results are parsed from training log output.
# NCCL_IB_DISABLE=1 is set for compatibility.
# ============================================================================

BASELINE_BPB="1.2244"
SEEDS=(1337 42 7 2024 31415)

# Parse arguments or use environment variables
DATA_PATH="${1:-${DATA_PATH:-}}"
TOKENIZER_PATH="${2:-${TOKENIZER_PATH:-}}"
NUM_SEEDS="${3:-${NUM_SEEDS:-5}}"
NPROC="${NPROC:-8}"

if [[ -z "$DATA_PATH" ]]; then
    echo "ERROR: DATA_PATH must be set via environment or first argument"
    echo "Usage: $0 <DATA_PATH> <TOKENIZER_PATH> [NUM_SEEDS]"
    exit 1
fi

if [[ -z "$TOKENIZER_PATH" ]]; then
    echo "ERROR: TOKENIZER_PATH must be set via environment or second argument"
    echo "Usage: $0 <DATA_PATH> <TOKENIZER_PATH> [NUM_SEEDS]"
    exit 1
fi

if (( NUM_SEEDS > ${#SEEDS[@]} )); then
    echo "ERROR: NUM_SEEDS ($NUM_SEEDS) exceeds available seeds (${#SEEDS[@]})"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"
RESULTS_FILE="${SCRIPT_DIR}/multi_seed_results.txt"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: train_gpt.py not found at $TRAIN_SCRIPT"
    exit 1
fi

echo "============================================"
echo "Multi-Seed Runner: DepthRecurrentQAT"
echo "============================================"
echo "DATA_PATH:      $DATA_PATH"
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
echo "NUM_SEEDS:      $NUM_SEEDS"
echo "NPROC:          $NPROC"
echo "BASELINE_BPB:   $BASELINE_BPB"
echo "SEEDS:          ${SEEDS[*]:0:$NUM_SEEDS}"
echo "============================================"
echo ""

declare -a BPB_VALUES=()
declare -a SEED_LIST=()

for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED="${SEEDS[$i]}"
    SEED_LIST+=("$SEED")
    RUN_ID="seed_${SEED}"
    LOG_FILE="${SCRIPT_DIR}/seed_${SEED}.log"

    echo "--- Run $((i + 1))/${NUM_SEEDS}: SEED=${SEED} ---"
    echo "Log: ${LOG_FILE}"

    NCCL_IB_DISABLE=1 \
    RUN_ID="${RUN_ID}" \
    DATA_PATH="${DATA_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    VOCAB_SIZE=1024 \
    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS=600 \
    TRAIN_LOG_EVERY=50 \
    VAL_LOSS_EVERY=200 \
    torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" 2>&1 | tee "${LOG_FILE}"

    # Parse val_bpb from the final roundtrip metric line
    VAL_BPB=$(grep "final_int8_zlib_roundtrip_exact" "${LOG_FILE}" | tail -1 | sed -n 's/.*val_bpb:\([0-9.]*\).*/\1/p')

    if [[ -z "$VAL_BPB" ]]; then
        echo "WARNING: Could not parse val_bpb from seed ${SEED} log"
        VAL_BPB="NaN"
    fi

    BPB_VALUES+=("$VAL_BPB")
    echo "SEED=${SEED} val_bpb=${VAL_BPB}"
    echo ""
done

echo "============================================"
echo "Results Summary"
echo "============================================"

# Compute mean and standard deviation using awk
STATS=$(printf '%s\n' "${BPB_VALUES[@]}" | awk '
    /^[0-9]/ {
        sum += $1
        sumsq += $1 * $1
        n++
        vals[n] = $1
    }
    END {
        if (n == 0) { print "ERROR: no valid BPB values"; exit 1 }
        mean = sum / n
        if (n > 1) {
            var = (sumsq - sum * sum / n) / (n - 1)
            sd = sqrt(var)
            se = sd / sqrt(n)
        } else {
            sd = 0
            se = 0
        }
        printf "n=%d mean=%.8f sd=%.8f se=%.8f\n", n, mean, sd, se
        for (i = 1; i <= n; i++) printf "val_%d=%.8f\n", i, vals[i]
    }
')

N_VALID=$(echo "$STATS" | head -1 | sed 's/.*n=\([0-9]*\).*/\1/')
MEAN_BPB=$(echo "$STATS" | head -1 | sed 's/.*mean=\([0-9.]*\).*/\1/')
SD_BPB=$(echo "$STATS" | head -1 | sed 's/.*sd=\([0-9.]*\).*/\1/')
SE_BPB=$(echo "$STATS" | head -1 | sed 's/.*se=\([0-9.]*\).*/\1/')

echo "Individual BPB per seed:"
for i in $(seq 0 $((${#SEED_LIST[@]} - 1))); do
    echo "  SEED=${SEED_LIST[$i]}  val_bpb=${BPB_VALUES[$i]}"
done
echo ""
echo "Mean BPB:    ${MEAN_BPB}"
echo "Std Dev:     ${SD_BPB}"
echo "Std Error:   ${SE_BPB}"
echo "Baseline:    ${BASELINE_BPB}"
echo ""

# Check if improvement exceeds threshold
IMPROVEMENT=$(awk "BEGIN { printf \"%.8f\", ${BASELINE_BPB} - ${MEAN_BPB} }")
EXCEEDS=$(awk "BEGIN { print (${IMPROVEMENT} >= 0.005) ? \"YES\" : \"NO\" }")
echo "Improvement: ${IMPROVEMENT} (baseline - mean)"
echo "Exceeds 0.005 threshold: ${EXCEEDS}"
echo ""
echo "Statistical significance note:"
echo "  For p < 0.01 with 5 seeds, the improvement must exceed ~3.4x"
echo "  the standard error (one-sided t-test, df=4)."
echo "  Required improvement > 3.4 * ${SE_BPB} = $(awk "BEGIN { printf \"%.8f\", 3.4 * ${SE_BPB} }")"
IS_SIGNIFICANT=$(awk "BEGIN { print (${IMPROVEMENT} > 3.4 * ${SE_BPB} && ${SE_BPB} > 0) ? \"LIKELY\" : \"INSUFFICIENT\" }")
echo "  Significance: ${IS_SIGNIFICANT}"

# Save results to file
{
    echo "============================================"
    echo "Multi-Seed Results: DepthRecurrentQAT"
    echo "Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "============================================"
    echo ""
    echo "Seeds: ${SEED_LIST[*]}"
    echo "Baseline BPB: ${BASELINE_BPB}"
    echo ""
    echo "Individual results:"
    for i in $(seq 0 $((${#SEED_LIST[@]} - 1))); do
        echo "  SEED=${SEED_LIST[$i]}  val_bpb=${BPB_VALUES[$i]}"
    done
    echo ""
    echo "Mean BPB:         ${MEAN_BPB}"
    echo "Std Dev:          ${SD_BPB}"
    echo "Std Error:        ${SE_BPB}"
    echo "Improvement:      ${IMPROVEMENT}"
    echo "Exceeds 0.005:    ${EXCEEDS}"
    echo "Significance:     ${IS_SIGNIFICANT}"
    echo ""
    echo "For p < 0.01 with 5 seeds (one-sided t-test, df=4):"
    echo "  t_critical = 3.747 (exact), using 3.4 as conservative threshold"
    echo "  Required: improvement > 3.4 * SE = $(awk "BEGIN { printf \"%.8f\", 3.4 * ${SE_BPB} }")"
} > "${RESULTS_FILE}"

echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo "============================================"
