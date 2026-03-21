#!/usr/bin/env bash
# QAT Ablation — isolate QAT's effect on post-quantization val_bpb
#
# 4 runs, one variable (QAT on/off), two eval modes:
#   1. Baseline (no QAT, standard eval)      — reproduces naive baseline
#   2. Baseline (no QAT, sliding eval)       — reproduces SlidingWindowEval entry
#   3. QAT (sliding eval)                    — measures QAT's contribution
#   4. QAT (sliding eval, doc-isolated)      — measures doc isolation on top
#
# Architecture: 9L×512d, default hyperparams throughout.
# No FP16 embed, no warmdown tuning, no leader hyperparams.
#
# Usage:
#   cd /workspace/parameter-golf
#   bash records/track_10min_16mb/2026-03-19_QAT_Ablation/run_ablation.sh

set -euo pipefail

SCRIPT="records/track_10min_16mb/2026-03-19_QAT_Ablation/train_gpt.py"

# Baseline architecture + training — all defaults
BASE="VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4"
BASE="$BASE MLP_MULT=2 TIE_EMBEDDINGS=1 TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024"
BASE="$BASE ITERATIONS=20000 WARMDOWN_ITERS=1200 WARMUP_STEPS=20"
BASE="$BASE MAX_WALLCLOCK_SECONDS=600 TRAIN_LOG_EVERY=200 VAL_LOSS_EVERY=0"
BASE="$BASE NUM_LOOPS=1 LORA_RANK=0 FP16_EMBED_EXPORT=0 SEED=1337"

echo "============================================"
echo "QAT Ablation — 4 runs, 8×H100"
echo "============================================"

# Run 1: Baseline — no QAT, standard eval (non-overlapping)
echo ""
echo ">>> Run 1/4: Baseline (no QAT, standard eval)"
env $BASE RUN_ID=ablation_baseline QAT=0 EVAL_STRIDE=0 DOC_ISOLATED_EVAL=0 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT"
cp logs/ablation_baseline.txt records/track_10min_16mb/2026-03-19_QAT_Ablation/logs/ 2>/dev/null || true
echo ">>> Run 1 done: $(grep 'final_int8_zlib_roundtrip_exact' logs/ablation_baseline.txt | tail -1)"

# Run 2: No QAT, sliding eval (stride=64)
echo ""
echo ">>> Run 2/4: No QAT, sliding eval (stride=64)"
env $BASE RUN_ID=ablation_slide64 QAT=0 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 DOC_ISOLATED_EVAL=0 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT"
cp logs/ablation_slide64.txt records/track_10min_16mb/2026-03-19_QAT_Ablation/logs/ 2>/dev/null || true
echo ">>> Run 2 done: $(grep 'final_int8_zlib_roundtrip_exact' logs/ablation_slide64.txt | tail -1)"

# Run 3: QAT + sliding eval (stride=64)
echo ""
echo ">>> Run 3/4: QAT + sliding eval (stride=64)"
env $BASE RUN_ID=ablation_qat_slide64 QAT=1 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 DOC_ISOLATED_EVAL=0 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT"
cp logs/ablation_qat_slide64.txt records/track_10min_16mb/2026-03-19_QAT_Ablation/logs/ 2>/dev/null || true
echo ">>> Run 3 done: $(grep 'final_int8_zlib_roundtrip_exact' logs/ablation_qat_slide64.txt | tail -1)"

# Run 4: QAT + sliding eval + doc-isolated
echo ""
echo ">>> Run 4/4: QAT + sliding eval + doc-isolated"
env $BASE RUN_ID=ablation_qat_slide64_dociso QAT=1 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 DOC_ISOLATED_EVAL=1 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT"
cp logs/ablation_qat_slide64_dociso.txt records/track_10min_16mb/2026-03-19_QAT_Ablation/logs/ 2>/dev/null || true
echo ">>> Run 4 done: $(grep 'final_int8_zlib_roundtrip_exact' logs/ablation_qat_slide64_dociso.txt | tail -1)"

echo ""
echo "============================================"
echo "ABLATION RESULTS"
echo "============================================"
for LOG in ablation_baseline ablation_slide64 ablation_qat_slide64 ablation_qat_slide64_dociso; do
  echo "$LOG: $(grep 'final_int8_zlib_roundtrip_exact' logs/${LOG}.txt 2>/dev/null | tail -1)"
done
echo ""
echo "Expected pattern:"
echo "  baseline          ~1.2244  (reproduces naive baseline)"
echo "  slide64           ~1.1925  (reproduces SlidingWindowEval entry)"
echo "  qat+slide64       < slide64 if QAT helps"
echo "  qat+slide64+doc   < qat+slide64 if doc isolation helps"
