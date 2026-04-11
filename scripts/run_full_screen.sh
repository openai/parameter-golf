#!/bin/bash
# Full screening: R1 → R2 → R3, all sequential, with smoke tests.
# Usage: bash scripts/run_full_screen.sh
set -e
cd /workspace/parameter-golf
mkdir -p logs

# ---- GPU config ----
detect_max_gpus() {
    local available=$(nvidia-smi -L 2>/dev/null | wc -l)
    for n in 8 4 2 1; do
        if [ "$available" -ge "$n" ]; then echo "$n"; return; fi
    done
    echo "1"
}
NGPU=$(detect_max_gpus)
echo "Using $NGPU GPUs (grad_accum=$((8 / NGPU)))"

COMMON="DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024"
R1_BASE="NUM_LAYERS=11 MLP_MULT=3 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1"

run() {
    local name=$1
    local script=$2
    shift 2
    echo "=========================================="
    echo "RUNNING: $name ($(date))"
    echo "=========================================="
    eval "RUN_ID=${name} ${COMMON} $@ torchrun --standalone --nproc_per_node=${NGPU} ${script}" 2>&1
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "FAILED: $name (exit code $exit_code)"
    else
        echo "COMPLETED: $name ($(date))"
    fi
    echo ""
    return $exit_code
}

smoke() {
    local name=$1
    local script=$2
    shift 2
    echo "--- SMOKE TEST: $name ---"
    eval "RUN_ID=smoke_${name} ${COMMON} ITERATIONS=3 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=120 $@ torchrun --standalone --nproc_per_node=${NGPU} ${script}" 2>&1 | tail -3
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "SMOKE FAILED: $name — SKIPPING"
        return 1
    fi
    echo "SMOKE OK: $name"
    echo ""
    return 0
}

# =============================================
# PHASE 0: SMOKE TEST ALL SCRIPTS
# =============================================
echo "=========================================="
echo "PHASE 0: SMOKE TESTS ($(date))"
echo "=========================================="

SMOKE_FAIL=0

# R1 smoke
smoke "r1_base" "train_gpt_r1.py" $R1_BASE || SMOKE_FAIL=1

# R2 smoke — one per MLP type + data augmentation
smoke "r2_fan" "train_gpt_r2.py" $R1_BASE MLP_TYPE=fan || SMOKE_FAIL=1
smoke "r2_dml_gated" "train_gpt_r2.py" $R1_BASE MLP_TYPE=dml_gated BT_LAMBDA=0.01 || SMOKE_FAIL=1
smoke "r2_dml_orth" "train_gpt_r2.py" $R1_BASE MLP_TYPE=dml_orth || SMOKE_FAIL=1
smoke "r2_fan_dml" "train_gpt_r2.py" $R1_BASE MLP_TYPE=fan_dml BT_LAMBDA=0.01 || SMOKE_FAIL=1
smoke "r2_causal_wide" "train_gpt_r2.py" NUM_LAYERS=8 MLP_MULT=5 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1 MLP_TYPE=causal_wide BT_LAMBDA=0.01 || SMOKE_FAIL=1
smoke "r2_dml_causal_wide" "train_gpt_r2.py" NUM_LAYERS=8 MLP_MULT=5 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1 MLP_TYPE=dml_causal_wide BT_LAMBDA=0.01 || SMOKE_FAIL=1
smoke "r2_token_drop" "train_gpt_r2.py" $R1_BASE TOKEN_DROP_RATE=0.1 || SMOKE_FAIL=1
smoke "r2_corrupt" "train_gpt_r2.py" $R1_BASE CORRUPT_RATE=0.1 || SMOKE_FAIL=1

# R3 smoke
smoke "r3_sliding" "train_gpt_r3.py" $R1_BASE EVAL_STRIDE=64 || SMOKE_FAIL=1
smoke "r3_ttt" "train_gpt_r3.py" $R1_BASE TTT=1 TTT_LR=0.002 TTT_EPOCHS=1 TTT_FREEZE_BLOCKS=2 TTT_CHUNK_TOKENS=4096 || SMOKE_FAIL=1

if [ $SMOKE_FAIL -ne 0 ]; then
    echo "=========================================="
    echo "SOME SMOKE TESTS FAILED — check logs above"
    echo "Continuing with experiments that passed..."
    echo "=========================================="
fi

# Clean up smoke logs
rm -f logs/smoke_*.txt

# =============================================
# PHASE 1: R1 EXPERIMENTS
# =============================================
echo ""
echo "=========================================="
echo "PHASE 1: R1 EXPERIMENTS ($(date))"
echo "=========================================="

run "r1_1_leakyrelu_11L_3x" "train_gpt_r1.py" NUM_LAYERS=11 MLP_MULT=3 || true
run "r1_2_bigram" "train_gpt_r1.py" NUM_LAYERS=11 MLP_MULT=3 BIGRAM_VOCAB_SIZE=3072 || true
run "r1_3_xsa" "train_gpt_r1.py" NUM_LAYERS=11 MLP_MULT=3 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 || true
run "r1_5_full" "train_gpt_r1.py" $R1_BASE || true

# =============================================
# PHASE 2: R2 EXPERIMENTS
# =============================================
echo ""
echo "=========================================="
echo "PHASE 2: R2 EXPERIMENTS ($(date))"
echo "=========================================="

# 2A: MLP variants
run "r2_1_fan" "train_gpt_r2.py" $R1_BASE MLP_TYPE=fan || true
run "r2_2_dml_gated" "train_gpt_r2.py" $R1_BASE MLP_TYPE=dml_gated BT_LAMBDA=0.01 || true
run "r2_3_dml_orth" "train_gpt_r2.py" $R1_BASE MLP_TYPE=dml_orth || true
run "r2_4_fan_dml" "train_gpt_r2.py" $R1_BASE MLP_TYPE=fan_dml BT_LAMBDA=0.01 || true
run "r2_13_causal_wide" "train_gpt_r2.py" NUM_LAYERS=8 MLP_MULT=5 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1 MLP_TYPE=causal_wide BT_LAMBDA=0.01 || true
run "r2_14_dml_causal_wide" "train_gpt_r2.py" NUM_LAYERS=8 MLP_MULT=5 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1 MLP_TYPE=dml_causal_wide BT_LAMBDA=0.01 || true

# 2B: Data augmentation
run "r2_5_token_drop" "train_gpt_r2.py" $R1_BASE TOKEN_DROP_RATE=0.1 || true
run "r2_8_grad_drop" "train_gpt_r2.py" $R1_BASE TOKEN_DROP_SCHEDULE=linear_decay TOKEN_DROP_MAX_RATE=0.2 || true
run "r2_11_corrupt" "train_gpt_r2.py" $R1_BASE CORRUPT_RATE=0.1 || true
run "r2_12_grad_corrupt" "train_gpt_r2.py" $R1_BASE CORRUPT_SCHEDULE=sine CORRUPT_MAX_RATE=0.2 || true

# =============================================
# PHASE 3: R3 EXPERIMENTS (on R1 best config)
# =============================================
echo ""
echo "=========================================="
echo "PHASE 3: R3 EXPERIMENTS ($(date))"
echo "=========================================="

run "r3_sliding" "train_gpt_r3.py" $R1_BASE EVAL_STRIDE=64 || true
run "r3_ttt" "train_gpt_r3.py" $R1_BASE TTT=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=2 || true

# =============================================
# FINAL SUMMARY
# =============================================
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE ($(date))"
echo "=========================================="
echo ""
echo "RESULTS SUMMARY:"
echo "=========================================="
for log in logs/r1_*.txt logs/r2_*.txt logs/r3_*.txt; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .txt)
    result=$(grep "final_int8_zlib_roundtrip_exact" "$log" 2>/dev/null | tail -1)
    if [ -n "$result" ]; then
        echo "$name: $result"
    else
        echo "$name: FAILED (no result)"
    fi
done

echo ""
echo "CHECKPOINTS:"
echo "=========================================="
for ckpt_dir in checkpoints/*/; do
    [ -d "$ckpt_dir" ] || continue
    name=$(basename "$ckpt_dir")
    if [ -f "${ckpt_dir}final_model.pt" ]; then
        size=$(du -sh "${ckpt_dir}final_model.pt" | cut -f1)
        echo "$name: ${ckpt_dir}final_model.pt ($size)"
    fi
done
