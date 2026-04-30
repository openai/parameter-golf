#!/bin/bash
# PROTEUS Feature Comparison on DGX Spark (GB10)
# Runs baseline vs PROTEUS features on sp1024 data
# Results saved to /tmp/spark_comparison_results.txt

set -e
cd ~/parameter-golf

RESULTS=/tmp/spark_comparison_results.txt
echo "=== PROTEUS Feature Comparison ===" > "$RESULTS"
echo "Started: $(date)" >> "$RESULTS"
echo "" >> "$RESULTS"

# Common settings for GB10 single-GPU research runs
COMMON="VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=0 ITERATIONS=1000 WARMUP_STEPS=10 TRAIN_BATCH_TOKENS=49152 VAL_BATCH_TOKENS=49152 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100 GPTQ_ENABLED=0 TORCH_COMPILE_DISABLE=1 SEED=42"

echo "=== RUN 1: Baseline (no PROTEUS features) ===" | tee -a "$RESULTS"
env $COMMON \
  PARALLEL_START_LAYER=0 \
  TTT_ENABLED=0 \
  SLOT_ENABLED=0 \
  N_INT6_MLP_LAYERS=11 \
  python3 -u train_gpt_1218_slot.py 2>&1 | tee /tmp/spark_run1_baseline.log
echo "" >> "$RESULTS"
grep -E "val_loss|val_bpb|Serialized|model_params|Total" /tmp/spark_run1_baseline.log >> "$RESULTS" 2>/dev/null
echo "--- Run 1 done ---" | tee -a "$RESULTS"
echo "" >> "$RESULTS"

echo "=== RUN 2: With parallel residuals (PARALLEL_START_LAYER=6) ===" | tee -a "$RESULTS"
env $COMMON \
  PARALLEL_START_LAYER=6 \
  TTT_ENABLED=0 \
  SLOT_ENABLED=0 \
  N_INT6_MLP_LAYERS=6 \
  python3 -u train_gpt_1218_slot.py 2>&1 | tee /tmp/spark_run2_parallel.log
echo "" >> "$RESULTS"
grep -E "val_loss|val_bpb|Serialized|model_params|Total" /tmp/spark_run2_parallel.log >> "$RESULTS" 2>/dev/null
echo "--- Run 2 done ---" | tee -a "$RESULTS"
echo "" >> "$RESULTS"

echo "=== RUN 3: Parallel + SLOT ===" | tee -a "$RESULTS"
env $COMMON \
  PARALLEL_START_LAYER=6 \
  TTT_ENABLED=0 \
  SLOT_ENABLED=1 \
  N_INT6_MLP_LAYERS=6 \
  python3 -u train_gpt_1218_slot.py 2>&1 | tee /tmp/spark_run3_parallel_slot.log
echo "" >> "$RESULTS"
grep -E "val_loss|val_bpb|Serialized|model_params|Total|slot" /tmp/spark_run3_parallel_slot.log >> "$RESULTS" 2>/dev/null
echo "--- Run 3 done ---" | tee -a "$RESULTS"
echo "" >> "$RESULTS"

echo "=== SUMMARY ===" | tee -a "$RESULTS"
echo "Finished: $(date)" >> "$RESULTS"
echo "" >> "$RESULTS"
echo "Run 1 (baseline):" >> "$RESULTS"
grep "val_bpb" /tmp/spark_run1_baseline.log | tail -1 >> "$RESULTS" 2>/dev/null
echo "Run 2 (parallel + INT5):" >> "$RESULTS"
grep "val_bpb" /tmp/spark_run2_parallel.log | tail -1 >> "$RESULTS" 2>/dev/null
echo "Run 3 (parallel + INT5 + SLOT):" >> "$RESULTS"
grep "val_bpb" /tmp/spark_run3_parallel_slot.log | tail -1 >> "$RESULTS" 2>/dev/null

echo ""
echo "=== Results saved to $RESULTS ==="
cat "$RESULTS"
