#!/bin/bash
# =============================================================================
# Quantization Optimization Sweep — The Right Metric
#
# Instead of optimizing FP16 BPB, optimize POST-QUANTIZATION BPB.
# 6 runs on 8×H100. Total: ~$24.
#
# Usage: bash run_quant_optimization.sh [run_number]
# =============================================================================
set -e

RUN=${1:-help}

# Setup
cd /workspace/parameter-golf

echo "=== Quantization Optimization Sweep ==="
echo "Run: $RUN"
echo ""

# Common env vars for the #1 compliant stack (SP1024 since SP8192 data unavailable)
BASE_ENV="VOCAB_SIZE=1024 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model"

case $RUN in

# =============================================================================
# Run 1: SDClip k-sweep — re-quantize SAME model at different k values
# No retraining. Train once, quantize 8 times. ~15 minutes total.
# =============================================================================
1)
    echo "=== Run 1: Train base model, then sweep SDClip k ==="

    # First train the base model
    eval "RUN_ID=ksweep_base $BASE_ENV USE_ANS=1 torchrun --standalone --nproc_per_node=8 train_gpt_ans.py"

    # Now re-quantize at different k values
    # The model is saved at final_model.pt
    echo ""
    echo "=== SDClip k-sweep on trained model ==="
    python3 k_sweep.py --model final_model.pt
    ;;

# =============================================================================
# Run 2: ANS vs Brotli on THIS stack (not baseline)
# Same training, compare compression. ~12 minutes.
# =============================================================================
2)
    echo "=== Run 2: ANS vs Brotli comparison ==="

    # Brotli version
    eval "RUN_ID=brotli_test $BASE_ENV USE_ANS=0 torchrun --standalone --nproc_per_node=8 train_gpt_ans.py"
    echo "--- Brotli done ---"
    grep "final\|Serialized\|submission" logs/brotli_test.txt | tail -5

    echo ""

    # ANS version (same seed = same model, just different compression)
    eval "RUN_ID=ans_test $BASE_ENV USE_ANS=1 torchrun --standalone --nproc_per_node=8 train_gpt_ans.py"
    echo "--- ANS done ---"
    grep "final\|Serialized\|submission" logs/ans_test.txt | tail -5

    echo ""
    echo "=== Comparison ==="
    echo "Brotli:"
    grep "final_int6_sliding_window_exact\|Total submission" logs/brotli_test.txt | tail -2
    echo "ANS:"
    grep "final_int6_sliding_window_exact\|Total submission" logs/ans_test.txt | tail -2
    ;;

# =============================================================================
# Run 3: Train with higher weight decay (quantization-friendly)
# Higher WD → narrower weights → better quantization at lower k
# =============================================================================
3)
    echo "=== Run 3: Higher weight decay for quantization ==="
    eval "RUN_ID=highwd $BASE_ENV USE_ANS=1 WEIGHT_DECAY=0.15 torchrun --standalone --nproc_per_node=8 train_gpt_ans.py"
    ;;

# =============================================================================
# Run 4: Post-TTT GPTQ calibration
# Calibrate quantizer on adapted weights, not training weights
# =============================================================================
4)
    echo "=== Run 4: Post-TTT GPTQ calibration ==="
    eval "RUN_ID=postttt $BASE_ENV USE_ANS=1 POST_TTT_GPTQ=1 torchrun --standalone --nproc_per_node=8 train_gpt_ans.py"
    ;;

# =============================================================================
# Run 5: Progressive recurrence K=4 + ANS
# More effective depth for free compute
# =============================================================================
5)
    echo "=== Run 5: Progressive recurrence K=4 ==="
    eval "RUN_ID=progrecur $BASE_ENV USE_ANS=1 PROGRESSIVE_RECUR=1 RECUR_MAX_K=4 RECUR_DETACH=1 RECUR_LAYERS=3,4,5 RECUR_START_STEP=500 torchrun --standalone --nproc_per_node=8 train_gpt_ans.py"
    ;;

# =============================================================================
# Run 6: Best combination of everything that worked
# =============================================================================
6)
    echo "=== Run 6: Best combination ==="
    eval "RUN_ID=best_combo $BASE_ENV USE_ANS=1 POST_TTT_GPTQ=1 PROGRESSIVE_RECUR=1 RECUR_MAX_K=4 RECUR_DETACH=1 RECUR_LAYERS=3,4,5 RECUR_START_STEP=500 torchrun --standalone --nproc_per_node=8 train_gpt_ans.py"
    ;;

# =============================================================================
# all: Run everything sequentially
# =============================================================================
all)
    for i in 1 2 3 4 5 6; do
        echo ""
        echo "########################################"
        echo "# Starting Run $i"
        echo "########################################"
        bash $0 $i
        echo ""
        echo "Run $i complete."
        echo ""
    done

    echo "=== ALL RUNS COMPLETE ==="
    echo ""
    echo "Results summary:"
    for f in logs/ksweep_base.txt logs/brotli_test.txt logs/ans_test.txt logs/highwd.txt logs/postttt.txt logs/progrecur.txt logs/best_combo.txt; do
        if [ -f "$f" ]; then
            NAME=$(basename $f .txt)
            BPB=$(grep "final_int6_sliding_window_exact" $f 2>/dev/null | grep -o "val_bpb:[0-9.]*" | cut -d: -f2)
            SIZE=$(grep "Total submission size" $f 2>/dev/null | tail -1 | grep -o "[0-9]* bytes" | head -1)
            echo "  $NAME: BPB=$BPB Size=$SIZE"
        fi
    done
    ;;

*)
    echo "Usage: bash run_quant_optimization.sh [1|2|3|4|5|6|all]"
    echo ""
    echo "  1: SDClip k-sweep (train once, quantize 8 times)"
    echo "  2: ANS vs Brotli comparison"
    echo "  3: Higher weight decay (quantization-friendly training)"
    echo "  4: Post-TTT GPTQ calibration"
    echo "  5: Progressive recurrence K=4"
    echo "  6: Best combination of everything"
    echo "  all: Run everything"
    ;;

esac
