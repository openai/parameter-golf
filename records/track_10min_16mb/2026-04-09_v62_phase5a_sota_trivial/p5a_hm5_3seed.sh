#!/usr/bin/env bash
# 3-seed training + eval for the winning variant (p5a_hm5)
# - s1337 already trained (in runs/v62_p5a_hm5_s1337)
# - s1338, s1339 sequential train (~10min each)
# - Then parallel eval stride=64 SLOT=100 for all 3 seeds on 3 GPUs

set -uo pipefail
SCRIPT=records/track_10min_16mb/2026-04-09_v62_phase5a_sota_trivial/train_gpt.py

train_one() {
  local seed="$1"
  RUN_NAME="v62_p5a_hm5_s${seed}"
  LOGDIR="logs/${RUN_NAME}"
  mkdir -p "$LOGDIR"
  if [[ -f "runs/${RUN_NAME}/model.rans.ptz" ]]; then
    echo "[s${seed}] already trained, skip"
    return
  fi
  echo "=== Training s${seed} ==="
  env \
    SEED="${seed}" BF16_WEIGHT=0 \
    MATRIX_LR=0.025 TIED_EMBED_LR=0.035 SCALAR_LR=0.025 \
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
    MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3 \
    TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 \
    ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=3500 \
    LZMA9_AFTER_RANS=1 \
    EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1 \
    QK_GAIN_INIT=5.0 MUON_EQ_R=1 \
    HIDDEN_MULT=5.0 \
    torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
      --train --v61 --h100 --ema 0.9965 --ema-type ema --swa \
      --seed "${seed}" --run-name "${RUN_NAME}" \
      --log-every 500 --val-every 0 --save-every 0 \
      --qk-gain 5.0 \
      --data-dir data/datasets/fineweb10B_sp1024 \
      --tokenizer data/tokenizers/fineweb_1024_bpe.model \
      2>&1 | tail -30 | tee "${LOGDIR}/train_tail.log"
  echo "[s${seed}] DONE"
}

# Train missing seeds sequentially (s1337 already done)
train_one 1338
train_one 1339

# Parallel eval all 3 seeds on GPU 0, 1, 2
echo ""
echo "=== Parallel eval 3 seeds stride=64 SLOT=100 ==="
pids=()
gpu=0
for seed in 1337 1338 1339; do
  CKPT="runs/v62_p5a_hm5_s${seed}/model.rans.ptz"
  LOGDIR="logs/v62_p5a_hm5_s${seed}"
  mkdir -p "$LOGDIR"
  if [[ ! -f "$CKPT" ]]; then
    echo "s${seed}: missing ckpt, skip"; continue
  fi
  CUDA_VISIBLE_DEVICES=$gpu env EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1 \
    QK_GAIN_INIT=5.0 MUON_EQ_R=1 HIDDEN_MULT=5.0 \
    nohup python -u "$SCRIPT" --eval --checkpoint "$CKPT" \
      --stride 64 --batch-seqs 32 --seq-len 1024 --compile \
      --slot-lr 0.1 --slot-steps 100 --slot-lr-min 0.001 \
      --data-dir data/datasets/fineweb10B_sp1024 \
      --tokenizer data/tokenizers/fineweb_1024_bpe.model \
      > "${LOGDIR}/eval_final.log" 2>&1 &
  pids+=($!)
  gpu=$((gpu + 1))
done
echo "Launched ${#pids[@]} evals on GPUs 0..$((gpu-1)), PIDs: ${pids[@]}"
wait "${pids[@]}" 2>/dev/null
echo "3-SEED EVAL DONE"

echo ""
echo "=== FINAL 3-seed Summary ==="
for seed in 1337 1338 1339; do
  b=$(grep -oP 'val_bpb:\s*\K[0-9.]+' "logs/v62_p5a_hm5_s${seed}/eval_final.log" 2>/dev/null | tail -1)
  printf "  seed %d:  bpb=%s\n" "$seed" "${b:-?}"
done
