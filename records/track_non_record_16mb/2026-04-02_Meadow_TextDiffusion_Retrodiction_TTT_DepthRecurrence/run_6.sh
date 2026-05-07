#!/bin/bash
# Orchestration script for the 6-run matched ablation (§3.2 of README.md).
#
# Override any path via environment variables before running:
#   SCRIPT_DIR   — directory containing train_cdm.py + train_ablation_runner.py
#                  (default: the directory this script lives in)
#   DATA_DIR     — directory containing fineweb_train_*.bin + fineweb_val_000000.bin
#                  (v4096 shards; default: ./data)
#   TOKENIZER    — path to bpe_v4096.model (default: ./bpe_v4096.model)
#   OUT_DIR      — where .npz and .lzma checkpoints land (default: ./out)
#   CKPT_DIR     — intermediate checkpoint directory (default: ./ckpt)
#   LOG_DIR      — training logs (default: ./logs)
#
# Example (exactly matching the run reported in the PR):
#   SCRIPT_DIR=./gcp \
#   DATA_DIR=./gv4096/data \
#   TOKENIZER=./gv4096/bpe_v4096.model \
#   bash run_6.sh
set -e

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
DATA_DIR="${DATA_DIR:-./data}"
TOKENIZER="${TOKENIZER:-./bpe_v4096.model}"
OUT_DIR="${OUT_DIR:-./out}"
CKPT_DIR="${CKPT_DIR:-./ckpt}"
LOG_DIR="${LOG_DIR:-./logs}"

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

run_one () {
  local tag=$1 layers=$2 dim=$3 bdim=$4 xsa=$5 weight=$6
  echo "================================================================"
  echo "== RUN: $tag  (L=$layers d=$dim cdm_w=$weight)"
  echo "================================================================"
  python3 "$SCRIPT_DIR/train_ablation_runner.py" \
    --train_script "$SCRIPT_DIR/train_cdm.py" \
    --num_layers $layers --model_dim $dim --vocab_size 4096 \
    --bigram_dim $bdim --xsa_last_n $xsa \
    --cdm_weight $weight \
    -- \
    --train_budget_secs 540 \
    --steps 9999 \
    --data_dir "$DATA_DIR" --tokenizer_path "$TOKENIZER" \
    --save_path "$OUT_DIR/${tag}.npz" \
    --save_int6_path "$OUT_DIR/${tag}_int6.lzma" \
    --checkpoint_dir "$CKPT_DIR/${tag}" \
    --val_every 500 --val_tokens 1000000 \
    > "$LOG_DIR/${tag}_train.log" 2>&1
  echo "  train done -> $LOG_DIR/${tag}_train.log"
  tail -5 "$LOG_DIR/${tag}_train.log"
}

# 5L runs (d=256, bigram=128, xsa_last_n=2)
run_one 5L_w0    5 256 128 2 0.0
run_one 5L_w03   5 256 128 2 0.3
run_one 5L_w1    5 256 128 2 1.0

# 11L runs (d=512, bigram=128, xsa_last_n=4)
run_one 11L_w0   11 512 128 4 0.0
run_one 11L_w03  11 512 128 4 0.3
run_one 11L_w1   11 512 128 4 1.0

echo "================================================================"
echo "== ALL 6 TRAINING RUNS DONE"
echo "================================================================"
ls -la "$OUT_DIR"/*.npz
