#!/bin/bash
# Benchmark: Run top SOTA submissions alongside our best configs.
# Usage: bash scripts/run_benchmark.sh
set -e
cd /workspace/parameter-golf
mkdir -p logs

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
# Find max valid GPU count (must divide 8)
for n in 8 4 2 1; do
    if [ "$NGPU" -ge "$n" ]; then NGPU=$n; break; fi
done
echo "Using $NGPU GPUs (grad_accum=$((8 / NGPU)))"

COMMON="DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024"

run() {
    local name=$1
    local script=$2
    shift 2
    echo "=========================================="
    echo "RUNNING: $name ($(date))"
    echo "=========================================="
    eval "RUN_ID=${name} ${COMMON} $@ torchrun --standalone --nproc_per_node=${NGPU} ${script}" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAILED: $name (exit code $rc)"
    else
        echo "COMPLETED: $name ($(date))"
    fi
    echo ""
    return $rc
}

# =============================================
# SOTA #1: AR Self-Gen GPTQ + XSA-all + BigramHash 3072
# val_bpb 1.1147 on leaderboard (80 shards, 8xH100)
# =============================================
SOTA1="records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py"
if [ -f "$SOTA1" ]; then
    run "sota1_gptq_xsa" "$SOTA1" || true
else
    echo "SKIP: SOTA #1 script not found at $SOTA1"
fi

# =============================================
# SOTA #2: LeakyReLU² + Legal TTT + Parallel Muon
# val_bpb 1.1194 on leaderboard (80 shards, 8xH100)
# =============================================
SOTA2="records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py"
if [ -f "$SOTA2" ]; then
    run "sota2_leakyrelu_ttt" "$SOTA2" || true
else
    echo "SKIP: SOTA #2 script not found at $SOTA2"
fi

# =============================================
# OUR BEST: R1-5 full stack + corrupted context + DML-Gated MLP
# =============================================
R1_BASE="NUM_LAYERS=11 MLP_MULT=3 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1"

# Our best data augmentation
run "ours_corrupt" "train_gpt_r2.py" $R1_BASE CORRUPT_RATE=0.1 || true

# Our best novel MLP
run "ours_dml_gated" "train_gpt_r2.py" $R1_BASE MLP_TYPE=dml_gated BT_LAMBDA=0.01 || true

# Our best combo: DML-Gated + corruption
run "ours_dml_corrupt" "train_gpt_r2.py" $R1_BASE MLP_TYPE=dml_gated BT_LAMBDA=0.01 CORRUPT_RATE=0.1 || true

# Our R1-5 baseline for comparison
run "ours_r1_baseline" "train_gpt_r1.py" $R1_BASE || true

# =============================================
# RESULTS
# =============================================
echo ""
echo "=========================================="
echo "BENCHMARK COMPLETE ($(date))"
echo "=========================================="
echo ""
echo "RESULTS (sorted):"
echo "=========================================="
results=""
for log in logs/sota*.txt logs/ours*.txt; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .txt)
    bpb=$(grep "final_int8_zlib_roundtrip_exact" "$log" 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
    if [ -n "$bpb" ]; then
        results="${results}${bpb} ${name}\n"
    else
        echo "$name: FAILED (no result)"
    fi
done
echo -e "$results" | sort -n | while read bpb name; do
    [ -z "$name" ] && continue
    echo "$name: val_bpb=$bpb"
done
