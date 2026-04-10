#!/usr/bin/env bash
# Train-only sweep (no eval) — all variants run sequential, eval done later in parallel.
# Each variant train: ~10 min (600s + 3min startup + save). 6 variants = ~60-80 min total.

set -uo pipefail

SCRIPT=records/track_10min_16mb/2026-04-09_v62_phase5a_sota_trivial/train_gpt.py

run_train() {
  local name="$1"; shift
  local extra_env="$1"; shift
  local qk_gain="${1:-5.0}"; shift || true
  echo "==================================================================="
  echo "[$name] train-only"
  echo "  extra_env: $extra_env  qk_gain: $qk_gain"
  echo "==================================================================="
  RUN_NAME="v62_${name}_s1337"
  LOGDIR="logs/v62_${name}_s1337"
  mkdir -p "$LOGDIR"

  if [[ -f "runs/${RUN_NAME}/model.pt" ]]; then
    echo "[$name] model.pt exists, SKIP"
    return
  fi

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
      --log-every 500 --val-every 0 --save-every 0 \
      --qk-gain "${qk_gain}" \
      --data-dir data/datasets/fineweb10B_sp1024 \
      --tokenizer data/tokenizers/fineweb_1024_bpe.model \
      2>&1 | tail -25 | tee "${LOGDIR}/train_tail.log"

  if [[ -f "runs/${RUN_NAME}/model.rans.ptz" ]]; then
    SIZE=$(stat -c%s "runs/${RUN_NAME}/model.rans.ptz")
    echo "[$name] DONE — ${SIZE} bytes"
  else
    echo "[$name] FAIL — no rans.ptz"
  fi
}

# p5a_bg4096 already training; SKIP (will short-circuit by existing model.pt check)
run_train "p5a_bg4096"        "BIGRAM_VOCAB=4096"
run_train "p5a_hm5"           "HIDDEN_MULT=5.0"
run_train "p5a_bg4096_hm5"    "BIGRAM_VOCAB=4096 HIDDEN_MULT=5.0"
run_train "p5a_bg8192"        "BIGRAM_VOCAB=8192"
run_train "p5a_nl12"          "NUM_LAYERS=12"
run_train "p5a_ve4"           "VE_LAYERS=7,8,9,10"

echo "TRAIN SWEEP COMPLETE"
ls -la runs/ | grep -E 'v62_p5a_' | head -20
