#!/usr/bin/env bash
# Phase 5a (confirmed winner: QK 5.0 + MuonEq-R + EMA 0.9965 + int6_tok PTQ)
# + Phase 4 architecture re-invest sweep.
#
# Known baseline: Phase 0 v61_slot_steps100_1146 seed 1337 = 1.148530
# Known p5a seed 1337 @ 38% stride=64 = 1.141106 (trend to ~1.141 final)
#
# Run from parameter-golf root.

set -uo pipefail

SCRIPT=records/track_10min_16mb/2026-04-09_v62_phase5a_sota_trivial/train_gpt.py

run_train_eval() {
  local name="$1"; shift
  local extra_env="$1"; shift
  echo "==================================================================="
  echo "[$name]"
  echo "  extra_env: $extra_env"
  echo "==================================================================="
  RUN_NAME="v62_${name}_s1337"
  LOGDIR="logs/v62_${name}_s1337"
  mkdir -p "$LOGDIR"

  CKPT_PT="runs/${RUN_NAME}/model.pt"
  if [[ -f "$CKPT_PT" ]]; then
    echo "[$name] checkpoint already exists, skipping training"
  else
    env \
      SEED=1337 BF16_WEIGHT=0 \
      MATRIX_LR=0.025 TIED_EMBED_LR=0.035 SCALAR_LR=0.025 \
      MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
      MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3 \
      TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 \
      ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=3500 \
      LZMA9_AFTER_RANS=1 \
      EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1 \
      QK_GAIN_INIT=5.0 MUON_EQ_R=1 \
      $extra_env \
      torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
        --train --v61 --h100 --ema 0.9965 --ema-type ema --swa \
        --seed 1337 --run-name "${RUN_NAME}" \
        --log-every 200 --val-every 0 --save-every 0 \
        --qk-gain 5.0 \
        --data-dir data/datasets/fineweb10B_sp1024 \
        --tokenizer data/tokenizers/fineweb_1024_bpe.model \
        2>&1 | tee "${LOGDIR}/train.log"
  fi

  CKPT="runs/${RUN_NAME}/model.rans.ptz"
  if [[ ! -f "$CKPT" ]]; then
    echo "[$name] ERROR: checkpoint not found, skipping eval"
    return
  fi
  echo "[$name] eval stride=64 SLOT=100"
  env EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1 QK_GAIN_INIT=5.0 MUON_EQ_R=1 $extra_env \
    python "$SCRIPT" --eval --checkpoint "$CKPT" \
      --stride 64 --batch-seqs 32 --seq-len 1024 --compile \
      --slot-lr 0.1 --slot-steps 100 --slot-lr-min 0.001 \
      --data-dir data/datasets/fineweb10B_sp1024 \
      --tokenizer data/tokenizers/fineweb_1024_bpe.model \
      2>&1 | tee "${LOGDIR}/eval.log"
  echo "[$name] result:"
  grep -E "val_bpb|Sliding Window" "${LOGDIR}/eval.log" | tail -3
}

# Variant B1: bg=4096 (Phase 4 — bigger BigramHash)
run_train_eval "p5a_bg4096" "BIGRAM_VOCAB=4096"

# Variant B2: hidden_mult 5.0 (Phase 4 — wider MLP)
run_train_eval "p5a_hm5" "HIDDEN_MULT=5.0"

# Variant B3: bg4096 + hm5 combo
run_train_eval "p5a_bg4096_hm5" "BIGRAM_VOCAB=4096 HIDDEN_MULT=5.0"

# Variant B4: ve_layers 4 (more VE coverage)
run_train_eval "p5a_ve4" "VE_LAYERS=7,8,9,10"

echo "ALL DONE"
