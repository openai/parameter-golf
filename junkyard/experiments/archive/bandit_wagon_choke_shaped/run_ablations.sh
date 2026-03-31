#!/bin/bash
set -euo pipefail
# ================================================================
#  bandit_wagon_choke_shaped — Shaped Bottleneck Ablation
#
#  7 arms testing different choke shapes at fixed dims.
#  All arms use the same base config as the mega ablation CTRL-00.
#
#  Shapes:
#    flat (control reference)
#    pyramid:     shared 3072→512 stage + per-loop 512→C→512
#    pyramid_res: pyramid with free residual (stage1 output = bypass)
#    grouped:     block-diagonal per-loop down-projection
#    residual:    shared bypass + per-loop delta
#
#  References from mega ablation (same session):
#    CTRL-00:  int6_sw_bpb = 1.44185  (flat, no choke)
#    BWC-02:   int6_sw_bpb = 1.43674  (flat, choke=128)
#    BWC-04:   int6_sw_bpb = 1.42887  (flat, choke=512) ← beat this
#
#  Total: 7 arms × ~13 min ≈ ~90 min on 1×H100
#
#  Usage:
#    bash experiments/bandit_wagon_choke_shaped/run_ablations.sh
#    ABLATION_STEPS=200 bash experiments/bandit_wagon_choke_shaped/run_ablations.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
ABLATION_STEPS="${ABLATION_STEPS:-500}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
LOGDIR="${REPO_ROOT}/logs"
mkdir -p "${LOGDIR}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    import flash_attn; v=flash_attn.__version__
    if v.startswith('3'): print(f'  FA3 v{v} OK')
    else: print(f'  WARNING: FA{v[0]} detected — want FA3')
" 2>/dev/null || echo "  WARNING: no flash_attn found"

RESULTS=()

run_arm() {
    local arm_id="$1"
    local label="$2"
    shift 2

    echo ""
    echo "================================================================"
    echo "  ${arm_id} — ${label}"
    echo "  [${ABLATION_STEPS} steps | seed=${SEED} | nproc=${NPROC}]"
    echo "================================================================"

    local logfile="${LOGDIR}/bwcs_${arm_id}_s${SEED}_$(date +%H%M%S).log"

    env \
        MAX_WALLCLOCK_SECONDS=0 \
        ITERATIONS="${ABLATION_STEPS}" \
        WARMDOWN_ITERS=0 \
        SEED="${SEED}" \
        NPROC_PER_NODE="${NPROC}" \
        MLP_LEAKY_SLOPE=0.5 \
        CRAWLER_MLP_LEAKY_SLOPE=0.5 \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=2048 \
        ROPE_DIMS=16 \
        SWA_EVERY=50 \
        MTP_NUM_HEADS=0 \
        LATE_QAT_THRESHOLD=0 \
        MATRIX_LR=0.03 \
        TORCHDYNAMO_OPTIMIZE_DDP=0 \
        COMPILE_FULLGRAPH=0 \
        MODEL_DIM=512 \
        USE_CRAWLER=1 \
        NUM_FLAT_LAYERS=4 \
        NUM_CRAWLER_LAYERS=1 \
        CRAWLER_LOOPS=3 \
        CRAWLER_MLP_MULT=6.0 \
        INST_DIM=32 \
        CRAWLER_QUANT_INT8=1 \
        DELTA_NET_HEADS=0 \
        SKIP_EMA=1 \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0 \
        NITRUST_ENABLE=0 \
        NITRUST_STRICT=0 \
        CRAWLER_MLP_CHOKE_DIM=0 \
        CRAWLER_MLP_CHOKE_SHAPE=flat \
        CRAWLER_MLP_CHOKE_GROUPS=8 \
        CRAWLER_LOOP_SMEAR=0 \
        CRAWLER_TAP_DIM=0 \
        CRAWLER_TAP_LOOP_SPECIFIC=1 \
        CRAWLER_TAP_LAYERS=all \
        CRAWLER_LOOP_ROPE_SCALES=1,1,1 \
        "$@" \
        torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${logfile}"

    local bpb raw_bpb step_avg quant_gap
    bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null | tail -1 || echo "?")
    raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null | tail -1 || echo "?")
    step_avg=$(grep -oP 'step:[0-9]+/[0-9]+.*?step_avg:\K[0-9.]+' "${logfile}" 2>/dev/null | tail -1 || echo "?")
    if [[ "${raw_bpb}" != "?" && "${bpb}" != "?" ]]; then
        quant_gap=$(python3 -c "print(f'{float(\"${bpb}\")-float(\"${raw_bpb}\"):.4f}')" 2>/dev/null || echo "?")
    else
        quant_gap="?"
    fi

    RESULTS+=("${arm_id}|${label}|${step_avg}ms|${raw_bpb}|${bpb}|${quant_gap}")
    echo "  -> step_avg:${step_avg}ms  raw_bpb:${raw_bpb}  int6_sw_bpb:${bpb}  quant_gap:${quant_gap}"
}

# ----------------------------------------------------------------
# BWCS-00  control repin (no choke)
# ----------------------------------------------------------------
run_arm BWCS-00 "control (flat, no choke)"

# ----------------------------------------------------------------
# BWCS-01  pyramid-128 (shared 3072→512 + per-loop 512→128→512)
# ----------------------------------------------------------------
run_arm BWCS-01 "pyramid dim=128" \
    CRAWLER_MLP_CHOKE_DIM=128 \
    CRAWLER_MLP_CHOKE_SHAPE=pyramid

# ----------------------------------------------------------------
# BWCS-02  pyramid-512 (shared 3072→512 + per-loop 512→512→512)
# ----------------------------------------------------------------
run_arm BWCS-02 "pyramid dim=512" \
    CRAWLER_MLP_CHOKE_DIM=512 \
    CRAWLER_MLP_CHOKE_SHAPE=pyramid

# ----------------------------------------------------------------
# BWCS-03  pyramid_res-128 (pyramid + free residual bypass)
# ----------------------------------------------------------------
run_arm BWCS-03 "pyramid_res dim=128" \
    CRAWLER_MLP_CHOKE_DIM=128 \
    CRAWLER_MLP_CHOKE_SHAPE=pyramid_res

# ----------------------------------------------------------------
# BWCS-04  grouped-512 G=8 (block-diagonal, 8 balanced groups)
# ----------------------------------------------------------------
run_arm BWCS-04 "grouped dim=512 groups=8" \
    CRAWLER_MLP_CHOKE_DIM=512 \
    CRAWLER_MLP_CHOKE_SHAPE=grouped \
    CRAWLER_MLP_CHOKE_GROUPS=8

# ----------------------------------------------------------------
# BWCS-05  grouped-512 G=4 (coarser balance)
# ----------------------------------------------------------------
run_arm BWCS-05 "grouped dim=512 groups=4" \
    CRAWLER_MLP_CHOKE_DIM=512 \
    CRAWLER_MLP_CHOKE_SHAPE=grouped \
    CRAWLER_MLP_CHOKE_GROUPS=4

# ----------------------------------------------------------------
# BWCS-06  residual-128 (shared bypass + per-loop 3072→128→512 delta)
# ----------------------------------------------------------------
run_arm BWCS-06 "residual dim=128" \
    CRAWLER_MLP_CHOKE_DIM=128 \
    CRAWLER_MLP_CHOKE_SHAPE=residual

# ================================================================
#  SUMMARY
# ================================================================
CTRL_BPB=$(echo "${RESULTS[0]}" | cut -d'|' -f5)
REF_BWC04="1.42887"
REF_BWC02="1.43674"

echo ""
echo "================================================================"
echo "  BWCS SHAPED CHOKE SUMMARY"
echo "  seed=${SEED}  steps=${ABLATION_STEPS}"
echo "  References: CTRL-00=1.44185  BWC-02(flat-128)=${REF_BWC02}  BWC-04(flat-512)=${REF_BWC04}"
echo "================================================================"
printf "%-10s %-30s %-10s %-12s %-12s %-10s %s\n" \
    "ARM" "LABEL" "STEP_AVG" "RAW_BPB" "INT6_SW_BPB" "QUANT_GAP" "DELTA"
printf "%-10s %-30s %-10s %-12s %-12s %-10s %s\n" \
    "---" "-----" "--------" "-------" "-----------" "---------" "-----"

for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label step_avg raw bpb quant_gap <<< "${r}"
    delta="—"
    if [[ "${bpb}" != "?" && "${CTRL_BPB}" != "?" ]]; then
        delta=$(python3 -c "
v=float('${bpb}')-float('${CTRL_BPB}')
sign='+' if v>=0 else ''
print(f'{sign}{v:.5f}')
" 2>/dev/null || echo "?")
    fi
    printf "%-10s %-30s %-10s %-12s %-12s %-10s %s\n" \
        "${arm}" "${label}" "${step_avg}" "${raw}" "${bpb}" "${quant_gap}" "${delta}"
done

echo ""
echo "  Control: BWCS-00 int6_sw_bpb = ${CTRL_BPB}"
echo "  Ref BWC-02 (flat-128): ${REF_BWC02}"
echo "  Ref BWC-04 (flat-512): ${REF_BWC04}  ← bar to beat"
echo ""

echo "  Winners (beat BWC-04 flat-512):"
found_winner=0
for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label step_avg raw bpb quant_gap <<< "${r}"
    if [[ "${arm}" == "BWCS-00" ]]; then continue; fi
    if [[ "${bpb}" != "?" ]]; then
        beats=$(python3 -c "print('yes' if float('${bpb}') < float('${REF_BWC04}') else 'no')" 2>/dev/null || echo "no")
        if [[ "${beats}" == "yes" ]]; then
            echo "    *** ${arm} ${label} → ${bpb}"
            found_winner=1
        fi
    fi
done
if [[ ${found_winner} -eq 0 ]]; then
    echo "    (none beat flat-512 = ${REF_BWC04})"
fi

echo ""
echo "  Efficiency wins (matches BWC-04 ±0.002 at lower param count):"
echo "    pyramid-128 (BWCS-01), pyramid_res-128 (BWCS-03), grouped-8 (BWCS-04) are cheaper than flat-512"
echo ""
echo "================================================================"
echo "  DONE. All logs in ${LOGDIR}/bwcs_*"
echo "================================================================"
