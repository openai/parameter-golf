#!/bin/bash
set -e
# exp8: SOTA + PHASED_TTT_PREFIX_DOCS=3500 (3 phases × 3500 = 10500 doc-evals).
# Hypothesis: more docs per phase while keeping 3 phases (4-phase was worse)
# gives better TTT adaptation than SOTA's 2500×3=7500.

SOTA_SCRIPT="records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/train_gpt.py"
LOG_DIR="logs/exp8_prefix3500"
SEEDS="${SEEDS:-42 0 1234}"

mkdir -p "$LOG_DIR"

for SEED in $SEEDS; do
  echo "=== Seed ${SEED} | $(date) ==="
  NCCL_NET=Socket \
  DATA_DIR=./data \
  VOCAB_SIZE=8192 \
  DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 \
  ITERATIONS=20000 \
  MAX_WALLCLOCK_SECONDS=600 \
  PHASED_TTT_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=3500 \
  PHASED_TTT_NUM_PHASES=3 \
  EMBED_BITS=7 \
  MATRIX_LR=0.026 \
  MIN_LR=0.1 \
  MLP_CLIP_SIGMAS=11.5 \
  ATTN_CLIP_SIGMAS=13.0 \
  EMBED_CLIP_SIGMAS=14.0 \
  GRAD_CLIP_NORM=0.3 \
  TTT_CHUNK_SIZE=48 \
  WARMUP_STEPS=20 \
  MUON_BACKEND_STEPS=5 \
  GLOBAL_TTT_MOMENTUM=0.9 \
  WARMDOWN_FRAC=0.85 \
  BETA2=0.99 \
  TTT_BETA2=0.99 \
  TTT_WEIGHT_DECAY=0.5 \
  TTT_LORA_RANK=80 \
  SPARSE_ATTN_GATE_SCALE=0.5 \
  GPTQ_RESERVE_SECONDS=0.5 \
  GPTQ_CALIBRATION_BATCHES=16 \
  VAL_LOSS_EVERY=0 \
  GATED_ATTN_QUANT_GATE=1 \
  SPARSE_ATTN_GATE_ENABLED=1 \
  GATE_WINDOW=12 \
  SMEAR_GATE_ENABLED=1 \
  LQER_ENABLED=1 \
  LQER_ASYM_ENABLED=1 \
  LQER_RANK=4 \
  LQER_FACTOR_BITS=4 \
  LQER_ASYM_GROUP=64 \
  LQER_TOP_K=3 \
  FUSED_CE_ENABLED=1 \
  COMPRESSOR=pergroup \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 "$SOTA_SCRIPT" \
    2>&1 | tee "$LOG_DIR/seed_${SEED}.log"

  echo "Seed ${SEED} done. $(date)"
done

echo ""
echo "=== exp8 complete ==="
for SEED in $SEEDS; do
  BPB=$(grep -oP 'quantized_ttt_phased.*val_bpb:\K[0-9.]+' "$LOG_DIR/seed_${SEED}.log" | tail -1)
  echo "  seed=${SEED}  val_bpb=${BPB}"
done
python3 -c "
import sys, re
seeds = '${SEEDS}'.split()
vals = []
for s in seeds:
    log = open('${LOG_DIR}/seed_' + s + '.log').read()
    m = re.findall(r'quantized_ttt_phased.*val_bpb:([0-9.]+)', log)
    if m: vals.append(float(m[-1]))
if vals: print(f'  mean={sum(vals)/len(vals):.8f}  n={len(vals)}')
"
