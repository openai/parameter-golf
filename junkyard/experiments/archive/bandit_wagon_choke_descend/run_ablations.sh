#!/bin/bash
set -euo pipefail
# ================================================================
#  BWCD — Descending Battery on Pyramid-512
#
#  BWB mega ablation showed 9,3,1 (descending) has near-zero
#  quant_gap (+0.0001) vs ascending 1,3,9 (+0.0028). Hypothesis:
#  wide-first → narrow is the natural refinement order for the
#  crawler. Loop 0 establishes context basin, loops 1+2 refine.
#
#  All arms: pyramid-512 choke (BWCS winner).
#
#  Arms:
#    BWCD-00: pyramid-512 + 9,3,1   descending (mirror of 1,3,9)
#    BWCD-01: pyramid-512 + 4,2,1   gentle descending (mirror of 1,2,4)
#    BWCD-02: pyramid-512 + 9,1,1   first loop wide only
#
#  References (BWCS run, same seed/steps):
#    BWCS-00 flat ctrl:   1.45760925
#    BWCS-02 pyramid-512: 1.44724192
#  References (BWCB run, ascending on pyramid):
#    BWCB-00 pyramid + 1,2,4: TBD
#    BWCB-01 pyramid + 1,3,9: TBD
#
#  Total: 3 arms × ~10 min ≈ ~30 min on 1×H100
#
#  Usage:
#    bash experiments/bandit_wagon_choke_descend/run_ablations.sh
#    ABLATION_STEPS=200 bash experiments/bandit_wagon_choke_descend/run_ablations.sh
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

    local logfile="${LOGDIR}/bwcd_${arm_id}_s${SEED}_$(date +%H%M%S).log"

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
        CRAWLER_MLP_CHOKE_SHAPE=pyramid \
        CRAWLER_MLP_CHOKE_GROUPS=8 \
        CRAWLER_MLP_CHOKE_DIM=512 \
        CRAWLER_LOOP_SMEAR=0 \
        CRAWLER_TAP_DIM=0 \
        CRAWLER_TAP_LOOP_SPECIFIC=1 \
        CRAWLER_TAP_LAYERS=all \
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
# BWCD-00  pyramid-512 + 9,3,1 (descending — mirror of 1,3,9)
# ----------------------------------------------------------------
run_arm BWCD-00 "pyramid-512 + rope 9,3,1 (descending)" \
    CRAWLER_LOOP_ROPE_SCALES=9,3,1

# ----------------------------------------------------------------
# BWCD-01  pyramid-512 + 4,2,1 (gentle descending — mirror of 1,2,4)
# ----------------------------------------------------------------
run_arm BWCD-01 "pyramid-512 + rope 4,2,1 (gentle desc)" \
    CRAWLER_LOOP_ROPE_SCALES=4,2,1

# ----------------------------------------------------------------
# BWCD-02  pyramid-512 + 9,1,1 (first loop wide only)
# ----------------------------------------------------------------
run_arm BWCD-02 "pyramid-512 + rope 9,1,1 (first wide)" \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1

# ----------------------------------------------------------------
# BWCD-03  pyramid-512 + 9,3,9 (wide-medium-wide bracket)
# Loop 0 establishes global context. Loop 1 refines structure.
# Loop 2 re-integrates globally on the already-refined residual.
# Loops 0+2 share the same causal horizon — outer loops bracket
# the medium refinement. Balance: open wide, refine, close wide.
# ----------------------------------------------------------------
run_arm BWCD-03 "pyramid-512 + rope 9,3,9 (wide-med-wide)" \
    CRAWLER_LOOP_ROPE_SCALES=9,3,9

# ================================================================
#  SUMMARY
# ================================================================
REF_CTRL="1.45760925"
REF_PYRAMID="1.44724192"

echo ""
echo "================================================================"
echo "  BWCD DESCENDING + BRACKET SUMMARY"
echo "  seed=${SEED}  steps=${ABLATION_STEPS}"
echo "  References:"
echo "    BWCS-00 flat ctrl:   ${REF_CTRL}"
echo "    BWCS-02 pyramid-512: ${REF_PYRAMID}  <- beat this"
echo "    BWB-04  flat 9,3,1:  1.44156  quant_gap=+0.0001"
echo "    BWB-01  flat 1,2,4:  1.43769  quant_gap=-0.0010"
echo "================================================================"
printf "%-10s %-38s %-10s %-12s %-12s %-10s %s\n" \
    "ARM" "LABEL" "STEP_AVG" "RAW_BPB" "INT6_SW_BPB" "QUANT_GAP" "vs PYRAMID"
printf "%-10s %-38s %-10s %-12s %-12s %-10s %s\n" \
    "---" "-----" "--------" "-------" "-----------" "---------" "----------"

for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label step_avg raw bpb quant_gap <<< "${r}"
    vs_pyramid="?"
    if [[ "${bpb}" != "?" ]]; then
        vs_pyramid=$(python3 -c "
v=float('${bpb}')-float('${REF_PYRAMID}')
sign='+' if v>=0 else ''
print(f'{sign}{v:.5f}')
" 2>/dev/null || echo "?")
    fi
    printf "%-10s %-38s %-10s %-12s %-12s %-10s %s\n" \
        "${arm}" "${label}" "${step_avg}" "${raw}" "${bpb}" "${quant_gap}" "${vs_pyramid}"
done

echo ""
echo "  Key question: does descending order preserve near-zero quant_gap"
echo "  from BWB-04 (flat 9,3,1) when combined with pyramid-512?"
echo "  If yes AND beats pyramid alone: descending is the correct battery order."
echo ""
echo "================================================================"
echo "  DONE. Logs in ${LOGDIR}/bwcd_*"
echo "================================================================"
