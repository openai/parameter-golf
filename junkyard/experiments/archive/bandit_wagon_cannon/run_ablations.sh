#!/bin/bash
set -euo pipefail
# ================================================================
#  BWE — Cannon Ablation: Per-Loop Output Calibration
#
#  The battery (9,1,1) aligned the attention side — what each loop
#  reads. The cannon aligns the output side — what each loop fires
#  into the residual stream for the next loop to receive.
#
#  Mechanism: applied to the DELTA (loop_out - loop_in), so the
#  cannon is a no-op at initialization and only grows away from 1.0
#  if the model finds it beneficial.
#
#  All arms: pyramid-512 + 9,1,1 (validated BWCD-02 config).
#
#  Arms:
#    BWE-00: control (no cannon) — must match BWCD-02 proxy
#    BWE-01: scalar — 1 learnable gain per loop (3 params)
#    BWE-02: channel — per-channel gain vector (3×512 = 1.5K params)
#    BWE-03: rmsnorm — RMSNorm on delta (3×512 = 1.5K params)
#
#  References:
#    BWCS-02 flat ctrl (1 shard): 1.45761
#    BWCS-02 pyramid-512 (1 shard): 1.44724
#    BWCD-02 pyramid + 9,1,1 (1 shard): 1.43531  ← bar to beat
#
#  Usage:
#    bash experiments/bandit_wagon_cannon/run_ablations.sh
#    ABLATION_STEPS=200 bash experiments/bandit_wagon_cannon/run_ablations.sh
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

    local logfile="${LOGDIR}/bwe_${arm_id}_s${SEED}_$(date +%H%M%S).log"

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
        CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
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
# BWE-00  control — pyramid-512 + 9,1,1, no cannon
# ----------------------------------------------------------------
run_arm BWE-00 "control (no cannon)" \
    CRAWLER_CANNON_TYPE=none

# ----------------------------------------------------------------
# BWE-01  scalar — 1 learnable gain per loop (3 params)
# ----------------------------------------------------------------
run_arm BWE-01 "scalar cannon (3 params)" \
    CRAWLER_CANNON_TYPE=scalar

# ----------------------------------------------------------------
# BWE-02  channel — per-channel gain vector per loop (3×512 = 1.5K)
# ----------------------------------------------------------------
run_arm BWE-02 "channel cannon (1.5K params)" \
    CRAWLER_CANNON_TYPE=channel

# ----------------------------------------------------------------
# BWE-03  rmsnorm — RMSNorm on delta per loop (3×512 = 1.5K)
# ----------------------------------------------------------------
run_arm BWE-03 "rmsnorm cannon (1.5K params)" \
    CRAWLER_CANNON_TYPE=rmsnorm

# ================================================================
#  SUMMARY
# ================================================================
REF_PYRAMID="1.44724192"
REF_BWCD02="1.43531057"

echo ""
echo "================================================================"
echo "  BWE CANNON SUMMARY"
echo "  seed=${SEED}  steps=${ABLATION_STEPS}"
echo "  References:"
echo "    BWCS-02 pyramid-512:     ${REF_PYRAMID}"
echo "    BWCD-02 pyramid + 9,1,1: ${REF_BWCD02}  <- bar to beat"
echo "================================================================"
printf "%-10s %-30s %-10s %-12s %-12s %-10s %s\n" \
    "ARM" "LABEL" "STEP_AVG" "RAW_BPB" "INT6_SW_BPB" "QUANT_GAP" "vs BWCD-02"
printf "%-10s %-30s %-10s %-12s %-12s %-10s %s\n" \
    "---" "-----" "--------" "-------" "-----------" "---------" "----------"

for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label step_avg raw bpb quant_gap <<< "${r}"
    vs_bwcd02="?"
    if [[ "${bpb}" != "?" ]]; then
        vs_bwcd02=$(python3 -c "
v=float('${bpb}')-float('${REF_BWCD02}')
sign='+' if v>=0 else ''
print(f'{sign}{v:.5f}')
" 2>/dev/null || echo "?")
    fi
    printf "%-10s %-30s %-10s %-12s %-12s %-10s %s\n" \
        "${arm}" "${label}" "${step_avg}" "${raw}" "${bpb}" "${quant_gap}" "${vs_bwcd02}"
done

echo ""
echo "  Cannon adds value if any arm beats BWCD-02 (1.43531)."
echo "  Watch: cannon[0] should diverge from 1.0 (loop 0 wide, diff amplitude)."
echo "  Watch: cannon[1] and cannon[2] should stay near 1.0 (loops 1+2 identical)."
echo ""
echo "================================================================"
echo "  DONE. Logs in ${LOGDIR}/bwe_*"
echo "================================================================"
