#!/bin/bash
set -euo pipefail
# BIOLOGY CONCEPTS SWEEP
# Runs all bio-inspired concept experiments vs green baseline.
# Usage:   bash experiments/Biology_concepts/run_all.sh
# H100:    WALLCLOCK=180 NPROC=8 bash experiments/Biology_concepts/run_all.sh
# Quick:   WALLCLOCK=60  NPROC=1 bash experiments/Biology_concepts/run_all.sh

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PATH="/home/frosty40/miniconda3/bin:${PATH}"

WALLCLOCK="${WALLCLOCK:-180}"
NPROC="${NPROC:-8}"
SEED="${SEED:-1337}"
LOG_DIR="${REPO_ROOT}/logs/bio_concepts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "  BIOLOGY CONCEPTS SWEEP"
echo "  wallclock=${WALLCLOCK}s per arm  |  gpus=${NPROC}  |  seed=${SEED}"
echo "  Concepts: tornado, theta_gamma, myelin, circadian, astrocyte, clonal_selection"
echo "  Logs: ${LOG_DIR}"
echo "========================================================"

declare -a NAMES=(   "baseline"       "tornado"        "theta_gamma"    "myelin"         "circadian"      "astrocyte"      "clonal_selection" )
declare -a SCRIPTS=( "experiments/baseline_run.sh"
                     "experiments/tornado/run.sh"
                     "experiments/theta_gamma/run.sh"
                     "experiments/myelin/run.sh"
                     "experiments/circadian/run.sh"
                     "experiments/astrocyte/run.sh"
                     "experiments/clonal_selection/run.sh" )

declare -a LOG_FILES=()

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    script="${SCRIPTS[$i]}"
    logfile="${LOG_DIR}/${name}.log"
    LOG_FILES+=("${logfile}")
    echo ""
    echo "--- ${name} ---"
    MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" NPROC_PER_NODE="${NPROC}" SEED="${SEED}" \
        bash "${script}" 2>&1 | tee "${logfile}"
    echo "    done -> ${logfile}"
done

echo ""
echo "========================================================"
echo "  RESULTS"
printf "%-20s  %-12s  %-12s  %s\n" "EXPERIMENT" "BASE_BPB" "NGRAM_BPB" "DELTA"
echo "------------------------------------------------------------"

baseline_bpb=""
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    logfile="${LOG_FILES[$i]}"

    base_bpb=$(grep -oP 'final_sliding_window_exact val_bpb:\K[\d.]+' "${logfile}" 2>/dev/null | tail -1 || echo "N/A")
    ngram_bpb=$(grep -oP 'final_sliding_window_ngram9_exact val_bpb:\K[\d.]+' "${logfile}" 2>/dev/null | tail -1 \
             || grep -oP 'final_sliding_window_ngram9_partial val_bpb:\K[\d.]+' "${logfile}" 2>/dev/null | tail -1 \
             || echo "N/A")

    if [ "${i}" -eq 0 ]; then
        baseline_bpb="${ngram_bpb}"
        delta="(baseline)"
    else
        if [ "${ngram_bpb}" != "N/A" ] && [ -n "${baseline_bpb}" ] && [ "${baseline_bpb}" != "N/A" ]; then
            delta=$(python3 -c "print(f'{float(\"${ngram_bpb}\") - float(\"${baseline_bpb}\"):+.4f}')" 2>/dev/null || echo "N/A")
        else
            delta="N/A"
        fi
    fi

    printf "%-20s  %-12s  %-12s  %s\n" "${name}" "${base_bpb}" "${ngram_bpb}" "${delta}"
done

echo "========================================================"
echo "  negative delta = improvement over green v1 baseline (1.1129 BPB SOTA)"
echo "========================================================"
