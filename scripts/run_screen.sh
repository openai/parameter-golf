#!/bin/bash
# Experiment screening runner — supports multiple GPU types.
#
# Usage:
#   ./scripts/run_screen.sh [OPTIONS] <round>
#
# Options:
#   --gpu rtx5090|h100    GPU type (default: rtx5090)
#   --gpus N              Override GPU count (default: auto-detect max valid)
#   --script FILE         Training script (default: train_gpt_r1.py for r1, train_gpt_r2.py for r2)
#   --dry-run             Print commands without executing
#
# Examples:
#   ./scripts/run_screen.sh r1                   # R1 on RTX 5090 (auto GPUs)
#   ./scripts/run_screen.sh --gpu h100 r1        # R1 on 8xH100
#   ./scripts/run_screen.sh --gpus 4 r2          # R2 on 4 GPUs
#   ./scripts/run_screen.sh --dry-run r2         # Preview R2 commands

set -e
cd /workspace/parameter-golf

# ---- Defaults ----
GPU_TYPE="rtx5090"
GPU_COUNT=""
SCRIPT=""
DRY_RUN=0
ROUND=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)    GPU_TYPE="$2"; shift 2 ;;
        --gpus)   GPU_COUNT="$2"; shift 2 ;;
        --script) SCRIPT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        r1|r2|r3) ROUND="$1"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$ROUND" ]; then
    echo "Usage: $0 [--gpu rtx5090|h100] [--gpus N] <r1|r2|r3>"
    exit 1
fi

# ---- GPU config ----
# grad_accum_steps = 8 // world_size, so world_size must divide 8
detect_max_gpus() {
    local available=$(nvidia-smi -L 2>/dev/null | wc -l)
    # Find largest divisor of 8 that is <= available
    for n in 8 4 2 1; do
        if [ "$available" -ge "$n" ]; then
            echo "$n"
            return
        fi
    done
    echo "1"
}

if [ -z "$GPU_COUNT" ]; then
    GPU_COUNT=$(detect_max_gpus)
fi

# Validate GPU_COUNT divides 8
if [ $((8 % GPU_COUNT)) -ne 0 ]; then
    echo "ERROR: GPU_COUNT=$GPU_COUNT does not divide 8. Use 1, 2, 4, or 8."
    exit 1
fi

GRAD_ACCUM=$((8 / GPU_COUNT))

# ---- Script selection ----
if [ -z "$SCRIPT" ]; then
    case "$ROUND" in
        r1) SCRIPT="train_gpt_r1.py" ;;
        r2) SCRIPT="train_gpt_r2.py" ;;
        r3) SCRIPT="train_gpt_r3.py" ;;
    esac
fi

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 1
fi

# ---- Common env ----
COMMON="DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024"

echo "=========================================="
echo "SCREENING CONFIG"
echo "  GPU type:    $GPU_TYPE"
echo "  GPU count:   $GPU_COUNT"
echo "  Grad accum:  $GRAD_ACCUM"
echo "  Script:      $SCRIPT"
echo "  Round:       $ROUND"
echo "=========================================="
echo ""

# ---- Runner ----
run() {
    local name=$1; shift
    echo "=========================================="
    echo "RUNNING: $name ($(date))"
    echo "=========================================="
    local cmd="RUN_ID=${name} ${COMMON} $@ torchrun --standalone --nproc_per_node=${GPU_COUNT} ${SCRIPT}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY RUN] $cmd"
    else
        eval "$cmd" 2>&1
    fi
    echo "COMPLETED: $name ($(date))"
    echo ""
}

# ---- R1 best config (used as base for R2) ----
R1_BASE="NUM_LAYERS=11 MLP_MULT=3 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1"

# ---- Experiment definitions ----
case "$ROUND" in
r1)
    run "r1_1_leakyrelu_11L_3x" NUM_LAYERS=11 MLP_MULT=3
    run "r1_2_bigram" NUM_LAYERS=11 MLP_MULT=3 BIGRAM_VOCAB_SIZE=3072
    run "r1_3_xsa" NUM_LAYERS=11 MLP_MULT=3 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4
    run "r1_5_full" NUM_LAYERS=11 MLP_MULT=3 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1
    ;;

r2)
    # 2A: MLP Architecture variants (on R1 best config)
    run "r2_1_fan"             $R1_BASE MLP_TYPE=fan
    run "r2_2_dml_gated"       $R1_BASE MLP_TYPE=dml_gated BT_LAMBDA=0.01
    run "r2_3_dml_orth"        $R1_BASE MLP_TYPE=dml_orth
    run "r2_4_fan_dml"         $R1_BASE MLP_TYPE=fan_dml BT_LAMBDA=0.01
    run "r2_13_causal_wide"    NUM_LAYERS=8 MLP_MULT=5 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1 MLP_TYPE=causal_wide BT_LAMBDA=0.01
    run "r2_14_dml_causal_wide" NUM_LAYERS=8 MLP_MULT=5 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1 MLP_TYPE=dml_causal_wide BT_LAMBDA=0.01

    # 2B: Data augmentation (on R1 best config, standard MLP)
    run "r2_5_token_drop"      $R1_BASE TOKEN_DROP_RATE=0.1
    run "r2_8_grad_drop"       $R1_BASE TOKEN_DROP_SCHEDULE=linear_decay TOKEN_DROP_MAX_RATE=0.2

    # 2B-extra: Corrupted context
    run "r2_11_corrupt"        $R1_BASE CORRUPT_RATE=0.1
    run "r2_12_grad_corrupt"   $R1_BASE CORRUPT_SCHEDULE=sine CORRUPT_MAX_RATE=0.2
    ;;

r3)
    # R3 on best R1+R2 config (to be updated after R2 screening)
    run "r3_sliding"           $R1_BASE EVAL_STRIDE=64
    run "r3_ttt"               $R1_BASE TTT=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=2
    ;;
esac

echo "=========================================="
echo "ALL $ROUND EXPERIMENTS COMPLETE ($(date))"
echo "=========================================="
echo ""
echo "RESULTS SUMMARY:"
echo "=========================================="
for log in logs/${ROUND}_*.txt; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .txt)
    result=$(grep "final_int8_zlib_roundtrip_exact" "$log" 2>/dev/null | tail -1 || echo "NO RESULT")
    echo "$name: $result"
done
