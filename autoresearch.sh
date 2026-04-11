#!/bin/bash
# Parameter Golf Autoresearch Loop
# Runs on PC1 (RTX 4090), iterates configs via env vars
# Each run = ~10min wall clock, logs results, picks next experiment
#
# Usage: bash autoresearch.sh
# Stop: kill the process or Ctrl+C

set -euo pipefail
cd ~/parameter-golf

RESULTS_FILE="autoresearch_results.tsv"
PYTHON=".venv/bin/python"
MAX_EXPERIMENTS=100
EXPERIMENT_NUM=0

# Initialize results file
if [ ! -f "$RESULTS_FILE" ]; then
    echo -e "experiment\tval_bpb\tval_bpb_int8_zlib\tval_bpb_ttt_lora\tsteps\tpeak_vram_mb\tstatus\tconfig" > "$RESULTS_FILE"
fi

# Default baseline config (matches train_gpt.py defaults)
DEFAULT_CONFIG=(
    "TRAIN_SEQ_LEN=1024"
    "NUM_LAYERS=9"
    "NUM_HEADS=8"
    "NUM_KV_HEADS=4"
    "MODEL_DIM=512"
    "MLP_MULT=2"
    "EMBED_LR=0.6"
    "MATRIX_LR=0.04"
    "SCALAR_LR=0.04"
    "TIED_EMBED_LR=0.05"
    "HEAD_LR=0.008"
    "MUON_MOMENTUM=0.95"
    "TTT_LORA_RANK=8"
    "TTT_LORA_LR=0.01"
    "TTT_EVAL_SEQ_LEN=1024"
    "QK_GAIN_INIT=1.5"
    "LOGIT_SOFTCAP=30.0"
    "ROPE_BASE=10000.0"
)

# Experiment queue - ordered by expected impact from leaderboard analysis
# Each entry: "name|ENV_VAR1=val ENV_VAR2=val ..."
EXPERIMENTS=(
    # Exp 0: Baseline (default config)
    "baseline|"

    # Exp 1: Double seq length (leaderboard #4 got 1.2014 with 4k)
    "seqlen_2048|TRAIN_SEQ_LEN=2048 TTT_EVAL_SEQ_LEN=2048"

    # Exp 2: More layers (leaderboard #1 used 10 layers)
    "layers_10|NUM_LAYERS=10"

    # Exp 3: Combine seqlen + layers
    "seqlen2048_layers10|TRAIN_SEQ_LEN=2048 TTT_EVAL_SEQ_LEN=2048 NUM_LAYERS=10"

    # Exp 4: Higher model dim
    "dim_640|MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5"

    # Exp 5: Aggressive seq length (4096)
    "seqlen_4096|TRAIN_SEQ_LEN=4096 TTT_EVAL_SEQ_LEN=4096"

    # Exp 6: Combine best architectural changes
    "seqlen2048_layers10_dim640|TRAIN_SEQ_LEN=2048 TTT_EVAL_SEQ_LEN=2048 NUM_LAYERS=10 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5"

    # Exp 7: LR tuning - higher matrix LR (Muon benefits from higher LR)
    "matrix_lr_06|MATRIX_LR=0.06 SCALAR_LR=0.06"

    # Exp 8: More MLP capacity
    "mlp_mult_3|MLP_MULT=3 MODEL_DIM=448 NUM_HEADS=8 NUM_KV_HEADS=4"

    # Exp 9: Higher TTT LoRA rank
    "ttt_rank_16|TTT_LORA_RANK=16 TTT_LORA_LR=0.005"

    # Exp 10: Combined winner (will be updated based on results)
    "combined_best|TRAIN_SEQ_LEN=2048 TTT_EVAL_SEQ_LEN=2048 NUM_LAYERS=10 MATRIX_LR=0.06 SCALAR_LR=0.06 TTT_LORA_RANK=16 TTT_LORA_LR=0.005"

    # Exp 11: Wider model with fewer layers
    "wide_shallow|MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 NUM_LAYERS=7"

    # Exp 12: QK gain tuning
    "qk_gain_2|QK_GAIN_INIT=2.0"

    # Exp 13: Logit softcap tuning
    "softcap_50|LOGIT_SOFTCAP=50.0"

    # Exp 14: Lower rope base (shorter range attention)
    "rope_5000|ROPE_BASE=5000.0"

    # Exp 15: Higher rope base (longer range attention)
    "rope_50000|ROPE_BASE=50000.0 TRAIN_SEQ_LEN=2048 TTT_EVAL_SEQ_LEN=2048"
)

run_experiment() {
    local name="$1"
    local env_overrides="$2"
    local log_file="logs/autoresearch_${EXPERIMENT_NUM}_${name}.log"

    echo "============================================="
    echo "Experiment $EXPERIMENT_NUM: $name"
    echo "Config: $env_overrides"
    echo "Log: $log_file"
    echo "Started: $(date)"
    echo "============================================="

    # Build env string - torch.compile OOMs on 4090 Triton shared mem, disable it
    local env_cmd="TORCH_COMPILE_DISABLE=1"
    if [ -n "$env_overrides" ]; then
        env_cmd="$env_cmd $env_overrides"
    fi

    # Run training
    local start_time=$(date +%s)
    eval "env $env_cmd $PYTHON train_gpt.py" > "$log_file" 2>&1
    local exit_code=$?
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    echo "Finished in ${elapsed}s (exit code: $exit_code)"

    # Parse results
    local val_bpb="0.000000"
    local val_bpb_int8="0.000000"
    local val_bpb_ttt="0.000000"
    local steps="0"
    local peak_vram="0"
    local status="crash"

    if [ $exit_code -eq 0 ]; then
        # Extract final val_bpb (last occurrence before int8/ttt lines)
        val_bpb=$(grep "^step:" "$log_file" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "0.000000")
        if [ "$val_bpb" = "0.000000" ]; then
            # Try wallclock line
            val_bpb=$(grep "wallclock" "$log_file" | grep -oP 'val_bpb:\K[0-9.]+' || echo "0.000000")
        fi
        val_bpb_int8=$(grep "final_int8_zlib_roundtrip " "$log_file" | grep -oP 'val_bpb:\K[0-9.]+' || echo "0.000000")
        val_bpb_ttt=$(grep "final_int8_ttt_lora " "$log_file" | grep -oP 'val_bpb:\K[0-9.]+' || echo "0.000000")
        steps=$(grep "^step:" "$log_file" | tail -1 | grep -oP 'step:\K[0-9]+' || echo "0")
        peak_vram=$(grep "peak_vram" "$log_file" | grep -oP '[0-9.]+' | tail -1 || echo "0")
        status="done"
    else
        # Check last 20 lines for error
        echo "--- CRASH LOG ---"
        tail -20 "$log_file"
        echo "--- END CRASH ---"
    fi

    # Log to results
    echo -e "${EXPERIMENT_NUM}_${name}\t${val_bpb}\t${val_bpb_int8}\t${val_bpb_ttt}\t${steps}\t${peak_vram}\t${status}\t${env_overrides:-defaults}" >> "$RESULTS_FILE"

    echo ""
    echo "Results: val_bpb=$val_bpb | int8_zlib=$val_bpb_int8 | ttt_lora=$val_bpb_ttt | steps=$steps"
    echo ""
}

# Main loop
mkdir -p logs

echo "========================================"
echo "Parameter Golf Autoresearch Loop"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Experiments queued: ${#EXPERIMENTS[@]}"
echo "Results file: $RESULTS_FILE"
echo "Started: $(date)"
echo "========================================"
echo ""

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name config <<< "$exp_entry"
    run_experiment "$name" "$config"
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
done

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"
echo ""
echo "Results summary:"
cat "$RESULTS_FILE"
echo ""
echo "Best result:"
tail -n +2 "$RESULTS_FILE" | sort -t$'\t' -k2 -n | head -1
