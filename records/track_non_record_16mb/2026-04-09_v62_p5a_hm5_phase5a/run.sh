#!/usr/bin/env bash
# 8xH100 RunPod execution script for v6.2 Phase 5a SOTA-trivial wins (p5a_hm5).
# Usage: bash run.sh <phase> <seed>
#   phase: train | eval | both   (default: both)
#   seed:  1337 | 1338 | 1339 ... (default: 1337)
# Must be run from the parameter-golf repo root.
#
# v6.2 Phase 5a stack (vs v6.1 1.146523 SLOT100 baseline):
#   1) QK_GAIN_INIT=5.0   (PR #1413)
#   2) MUON_EQ_R=1        (Muon Newton-Schulz row L2 normalize, PR #1394)
#   3) ema=0.9965         (PR #1421/#1445)
#   4) HIDDEN_MULT=5.0    (FFN dim 4×→5× re-investment)
#   5) EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1   (Phase 1A: int6 tied embedding)
#
# Training is the same 8×H100 / 600s wallclock recipe as v6.1 SLOT-100 (#1123 chain).
# Eval phase uses SLOT lr=0.1 steps=100 stride=64, identical to the v6.1 baseline.

set -euo pipefail

PHASE="${1:-both}"
SEED="${2:-1337}"
SCRIPT=records/track_non_record_16mb/2026-04-09_v62_p5a_hm5_phase5a/train_gpt.py
RUN_NAME="v62_p5a_hm5_s${SEED}"
LOGDIR="logs/${RUN_NAME}"
mkdir -p "$LOGDIR"

TRAIN_ENV=(
  SEED="${SEED}" BF16_WEIGHT=0
  MATRIX_LR=0.025 TIED_EMBED_LR=0.035 SCALAR_LR=0.025
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
  MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
  TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
  ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=3500
  LZMA9_AFTER_RANS=1
  EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1
  QK_GAIN_INIT=5.0 MUON_EQ_R=1
  HIDDEN_MULT=5.0
)

EVAL_ENV=(
  EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1
  QK_GAIN_INIT=5.0 MUON_EQ_R=1
  HIDDEN_MULT=5.0
)

if [[ "$PHASE" == "train" || "$PHASE" == "both" ]]; then
  echo "=== [v6.2 p5a_hm5] training seed=${SEED} ==="
  env "${TRAIN_ENV[@]}" \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
    --train --v61 --h100 --ema 0.9965 --ema-type ema --swa \
    --seed "${SEED}" --run-name "${RUN_NAME}" \
    --log-every 200 --val-every 0 --save-every 0 \
    --qk-gain 5.0 \
    --data-dir data/datasets/fineweb10B_sp1024 \
    --tokenizer data/tokenizers/fineweb_1024_bpe.model \
    2>&1 | tee "${LOGDIR}/train.log"
fi

if [[ "$PHASE" == "eval" || "$PHASE" == "both" ]]; then
  CKPT="runs/${RUN_NAME}/model.rans.ptz"
  [[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
  echo "=== [v6.2 p5a_hm5] evaluating ${CKPT} ==="
  env "${EVAL_ENV[@]}" \
  python "$SCRIPT" --eval --checkpoint "$CKPT" \
    --stride 64 --batch-seqs 32 --seq-len 1024 --compile \
    --slot --slot-lr 0.1 --slot-steps 100 --slot-lr-min 0.001 \
    --data-dir data/datasets/fineweb10B_sp1024 \
    --tokenizer data/tokenizers/fineweb_1024_bpe.model \
    2>&1 | tee "${LOGDIR}/eval.log"
  echo "=== eval done ==="
  grep -E "val_bpb|Sliding Window" "${LOGDIR}/eval.log" | tail -5
fi
