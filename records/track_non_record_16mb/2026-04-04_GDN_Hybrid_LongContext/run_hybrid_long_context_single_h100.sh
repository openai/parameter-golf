#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {8192|16384|32768}"
  exit 1
fi

SEQ_LEN="$1"

# GDN hybrid step times from compute_experiment.md (steady-state, steps 6-10):
#   8192:  ~899ms/step  → with 393216 batch (~0.75x default) ≈ 674ms → ~890 steps in 600s
#   16384: ~1037ms/step → same 393216 batch ≈ 1037ms         → ~579 steps in 600s
#   32768: ~1363ms/step → 524288 batch                        → ~440 steps in 600s
#
# Schedule ratios from baseline_long_context_single_h100.md:
#   MUON_MOMENTUM_WARMUP_STEPS ~= 0.18 * estimated_steps
#   WARMDOWN_ITERS             ~= 0.36 * estimated_steps

case "$SEQ_LEN" in
  8192)
    RUN_ID="${RUN_ID:-hybrid_gdn_1xh100_600s_seq8192}"
    TRAIN_BATCH_TOKENS=393216
    MUON_MOMENTUM_WARMUP_STEPS=160   # 0.18 * 890
    WARMDOWN_ITERS=320               # 0.36 * 890
    SEED_DEFAULT=1340
    ;;
  16384)
    RUN_ID="${RUN_ID:-hybrid_gdn_1xh100_600s_seq16384}"
    TRAIN_BATCH_TOKENS=393216
    MUON_MOMENTUM_WARMUP_STEPS=104   # 0.18 * 579
    WARMDOWN_ITERS=208               # 0.36 * 579
    SEED_DEFAULT=1341
    ;;
  32768)
    RUN_ID="${RUN_ID:-hybrid_gdn_1xh100_600s_seq32768}"
    TRAIN_BATCH_TOKENS=524288
    MUON_MOMENTUM_WARMUP_STEPS=79    # 0.18 * 440
    WARMDOWN_ITERS=158               # 0.36 * 440
    SEED_DEFAULT=1342
    ;;
  *)
    echo "Unsupported sequence length: $SEQ_LEN"
    echo "Expected one of: 8192, 16384, 32768"
    exit 1
    ;;
esac

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export TRAIN_SEQ_LEN="$SEQ_LEN"
export TRAIN_BATCH_TOKENS
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.030}"
export MATRIX_LR="${MATRIX_LR:-0.020}"
export SCALAR_LR="${SCALAR_LR:-0.020}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS
export WARMDOWN_ITERS
export SEED="${SEED:-$SEED_DEFAULT}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-50}"
export RUN_ID

# GDN hybrid arch defaults (from compute_experiment.md / train_gpt_gdn.py defaults):
#   MODEL_DIM=448, GDN_LAYERS=0,1,2,4,5,6,8 (7 GDN + 2 attn at layers 3,7),
#   GDN_HEAD_DIM_RATIO=0.75, GDN_EXPAND_V=2.0, GDN_CONV_SIZE=4
# No overrides needed — these are the script defaults.

echo "Running GDN hybrid with:"
echo "  RUN_ID=$RUN_ID"
echo "  TRAIN_SEQ_LEN=$TRAIN_SEQ_LEN"
echo "  TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS"
echo "  TIED_EMBED_LR=$TIED_EMBED_LR"
echo "  MATRIX_LR=$MATRIX_LR"
echo "  SCALAR_LR=$SCALAR_LR"
echo "  MUON_MOMENTUM=$MUON_MOMENTUM"
echo "  MUON_MOMENTUM_WARMUP_START=$MUON_MOMENTUM_WARMUP_START"
echo "  MUON_MOMENTUM_WARMUP_STEPS=$MUON_MOMENTUM_WARMUP_STEPS"
echo "  WARMDOWN_ITERS=$WARMDOWN_ITERS"
echo "  WARMUP_STEPS=$WARMUP_STEPS"
echo "  SEED=$SEED"
echo "  MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
echo "  (GDN arch: MODEL_DIM=448, GDN_LAYERS=0,1,2,4,5,6,8, HEAD_DIM_RATIO=0.75, EXPAND_V=2.0, CONV_SIZE=4)"

torchrun --standalone --nproc_per_node=1 "$(dirname "${BASH_SOURCE[0]}")/train_gpt_gdn.py"
