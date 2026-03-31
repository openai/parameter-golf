#!/bin/bash
set -euo pipefail
# CAMBRIAN BIO SEAM SWEEP
# Tests each bio seam controller in isolation on top of GDN (delta) base.
# Usage:   bash experiments/Cambrian/run_bio_sweep.sh
# H100:    WALLCLOCK=180 NPROC=8 bash experiments/Cambrian/run_bio_sweep.sh
# Quick:   WALLCLOCK=60  NPROC=1 bash experiments/Cambrian/run_bio_sweep.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PATH="/home/frosty40/miniconda3/bin:${PATH}"

WALLCLOCK="${WALLCLOCK:-180}"
NPROC="${NPROC:-8}"
SEED="${SEED:-1337}"
DELTA_LAYERS="${DELTA_LAYERS:-2}"
LOG_DIR="${REPO_ROOT}/logs/cambrian_bio_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "  CAMBRIAN BIO SEAM SWEEP"
echo "  wallclock=${WALLCLOCK}s per arm  |  gpus=${NPROC}  |  seed=${SEED}"
echo "  delta_layers=${DELTA_LAYERS}  |  seams: myelin circadian clonal astrocyte"
echo "  Logs: ${LOG_DIR}"
echo "========================================================"

# Each arm: name + bio seam flags (all others off)
declare -a NAMES=(
    "gdn_base"
    "gdn_myelin"
    "gdn_circadian"
    "gdn_clonal"
    "gdn_astrocyte"
    "gdn_all"
)
declare -a MYELIN=(     0 1 0 0 0 1 )
declare -a CIRCADIAN=(  0 0 1 0 0 1 )
declare -a CLONAL=(     0 0 0 1 0 1 )
declare -a ASTROCYTE=(  0 0 0 0 1 1 )

declare -a LOG_FILES=()

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    logfile="${LOG_DIR}/${name}.log"
    LOG_FILES+=("${logfile}")

    # Kill any lingering GPU workers from the previous arm
    pkill -f "train_gpt.py" 2>/dev/null || true
    sleep 3

    echo ""
    echo "--- ${name} (myelin=${MYELIN[$i]} circadian=${CIRCADIAN[$i]} clonal=${CLONAL[$i]} astrocyte=${ASTROCYTE[$i]}) ---"
    MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
    NPROC_PER_NODE="${NPROC}" \
    SEED="${SEED}" \
    CAMBRIAN_DELTA_LAYERS="${DELTA_LAYERS}" \
    CAMBRIAN_MYELIN="${MYELIN[$i]}" \
    CAMBRIAN_CIRCADIAN="${CIRCADIAN[$i]}" \
    CAMBRIAN_CLONAL="${CLONAL[$i]}" \
    CAMBRIAN_ASTROCYTE="${ASTROCYTE[$i]}" \
    SKIP_FINAL_EVAL=1 \
        bash "${SCRIPT_DIR}/run.sh" 2>&1 | tee "${logfile}" || true
    echo "    done -> ${logfile}"
done

echo ""
echo "========================================================"
echo "  RESULTS (vs gdn_base)"
printf "%-16s  %-12s  %-12s  %s\n" "ARM" "EMA_BPB" "TRAIN_BPB" "DELTA_vs_BASE"
echo "------------------------------------------------------------"

baseline_bpb=""
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    logfile="${LOG_FILES[$i]}"

    # SKIP_FINAL_EVAL=1: use DIAGNOSTIC post_ema val_bpb as final metric
    val_bpb=$(grep -oP 'DIAGNOSTIC post_ema val_bpb:\K[\d.]+' "${logfile}" 2>/dev/null | tail -1 || echo "N/A")
    train_bpb=$(grep -oP 'val_bpb:\K[\d.]+' "${logfile}" 2>/dev/null | grep -v '^[3-9]\.' | tail -1 || echo "N/A")

    if [ "${i}" -eq 0 ]; then
        baseline_bpb="${val_bpb}"
        delta="(baseline)"
    else
        if [ "${val_bpb}" != "N/A" ] && [ -n "${baseline_bpb}" ] && [ "${baseline_bpb}" != "N/A" ]; then
            delta=$(python3 -c "print(f'{float(\"${val_bpb}\") - float(\"${baseline_bpb}\"):+.4f}')" 2>/dev/null || echo "N/A")
        else
            delta="N/A"
        fi
    fi

    printf "%-16s  %-12s  %-12s  %s\n" "${name}" "${val_bpb}" "${train_bpb}" "${delta}"
done

echo "========================================================"
echo "  negative delta = seam improves over pure GDN baseline"
echo "========================================================"
