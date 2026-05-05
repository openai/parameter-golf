#!/usr/bin/env bash
# Sanity-first launcher: train baseline (no SOTA tricks) with only EMBED_QUANT_BITS=6
# to verify Phase 1-A int6_tok PTQ is harmless when applied to a normally-trained model.
#
# Then ablate Phase 5a tricks ONE AT A TIME on top of that baseline.
#
# Run from parameter-golf root.

set -uo pipefail   # NOT -e: failure of one variant must not abort the rest

SCRIPT=records/track_10min_16mb/2026-04-09_v62_phase5a_sota_trivial/train_gpt.py

run_train_eval() {
  local name="$1"; shift
  local extra_env="$1"; shift
  local extra_args="$1"; shift
  local ema_decay="$1"; shift
  echo "==================================================================="
  echo "[$name]"
  echo "  env: $extra_env"
  echo "  args: $extra_args"
  echo "  ema: $ema_decay"
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
      $extra_env \
      torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
        --train --v61 --h100 --ema "$ema_decay" --ema-type ema --swa \
        --seed 1337 --run-name "${RUN_NAME}" \
        --log-every 200 --val-every 0 --save-every 0 \
        ${extra_args} \
        --data-dir data/datasets/fineweb10B_sp1024 \
        --tokenizer data/tokenizers/fineweb_1024_bpe.model \
        2>&1 | tee "${LOGDIR}/train.log"
  fi

  CKPT="runs/${RUN_NAME}/model.rans.ptz"
  if [[ ! -f "$CKPT" ]]; then
    echo "[$name] ERROR: checkpoint not found, skipping eval"
    return
  fi
  echo "[$name] eval (stride=128 fast sanity + SLOT steps=100)"
  env EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1 $extra_env \
    python "$SCRIPT" --eval --checkpoint "$CKPT" \
      --stride 128 --batch-seqs 32 --seq-len 1024 --compile \
      --slot-lr 0.1 --slot-steps 100 --slot-lr-min 0.001 \
      --data-dir data/datasets/fineweb10B_sp1024 \
      --tokenizer data/tokenizers/fineweb_1024_bpe.model \
      2>&1 | tee "${LOGDIR}/eval.log"
  echo "[$name] result:"
  grep -E "val_bpb|Sliding Window" "${LOGDIR}/eval.log" | tail -3
}

# Variant A: baseline + int6_tok PTQ only (sanity, no SOTA tricks)
run_train_eval "p1a_int6tok" "" "--qk-gain 2.0" "0.997"

# Variant B: + EMA 0.9965 (smallest change)
run_train_eval "p1a_int6tok_ema9965" "" "--qk-gain 2.0" "0.9965"

# Variant C: + QK 5.0 (most suspicious)
run_train_eval "p1a_int6tok_qk5" "QK_GAIN_INIT=5.0" "--qk-gain 5.0" "0.997"

# Variant D: + MuonEq-R (also suspicious)
run_train_eval "p1a_int6tok_muoneqr" "MUON_EQ_R=1" "--qk-gain 2.0" "0.997"

echo "ALL DONE"
