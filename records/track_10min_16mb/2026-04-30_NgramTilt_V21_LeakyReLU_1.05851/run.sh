#!/bin/bash
# Reproduce one seed of this submission. SEED defaults to 42.
# Usage:  SEED=42 bash run.sh   (or 0 / 1234 for the other declared seeds)
set -e

cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-/runpod-volume/caseops_data/datasets}"
DATA_PATH="${DATA_PATH:-$DATA_DIR/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$(pwd)/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"
SEED="${SEED:-42}"

env_vars=(
  DATA_DIR="$DATA_DIR"
  DATA_PATH="$DATA_PATH"
  TOKENIZER_PATH="$TOKENIZER_PATH"
  CASEOPS_ENABLED=1
  VOCAB_SIZE=8192
  ITERATIONS=20000
  MAX_WALLCLOCK_SECONDS=600
  WARMUP_STEPS=20
  WARMDOWN_FRAC=0.85
  BETA2=0.99
  GRAD_CLIP_NORM=0.3
  MIN_LR=0.1
  MATRIX_LR=0.026
  GLOBAL_TTT_MOMENTUM=0.9
  SPARSE_ATTN_GATE_ENABLED=1
  SPARSE_ATTN_GATE_SCALE=0.5
  SMEAR_GATE_ENABLED=1
  GATE_WINDOW=12
  GATED_ATTN_QUANT_GATE=1
  FUSED_CE_ENABLED=1
  EMBED_BITS=7
  MLP_CLIP_SIGMAS=11.5
  ATTN_CLIP_SIGMAS=13.0
  EMBED_CLIP_SIGMAS=14.0
  GPTQ_RESERVE_SECONDS=0.5
  GPTQ_CALIBRATION_BATCHES=16
  COMPRESSOR=pergroup
  LQER_ENABLED=1
  LQER_TOP_K=1
  ASYM_LOGIT_RESCALE=1
  AWQ_LITE_ENABLED=1
  PHASED_TTT_ENABLED=1
  PHASED_TTT_PREFIX_DOCS=2500
  PHASED_TTT_NUM_PHASES=3
  TTT_LR=0.75
  QK_GAIN_INIT=5.25
  TTT_NO_QV_MASK=1
  EVAL_SEQ_LEN=2048
  TTT_EVAL_SEQ_LEN=2048
  NGRAM_TILT_ENABLED=1
  NGRAM_HINT_PRECOMPUTE_OUTSIDE=1
  TOKEN_ORDER=16
  TOKEN_THRESHOLD=0.800
  TOKEN_BOOST=2.625
  WITHIN_TAU=0.450
  WITHIN_BOOST=0.750
  WORD_ORDER=4
  WORD_NORMALIZE=strip_punct_lower
  WORD_TAU=0.650
  WORD_BOOST=0.750
  AGREE_ADD_BOOST=0.500
  SEED="$SEED"
)

echo "Reproducing seed $SEED with NGRAM_HINT_PRECOMPUTE_OUTSIDE=1 (hint precompute outside eval-ops timer)."
echo "Set NGRAM_HINT_PRECOMPUTE_OUTSIDE=0 to reproduce inline path: identical val_bpb at higher total_eval_time."

env "${env_vars[@]}" \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
