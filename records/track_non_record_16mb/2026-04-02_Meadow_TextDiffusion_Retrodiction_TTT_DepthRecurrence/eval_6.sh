#!/bin/bash
# CF evaluation orchestration for the 6-run matched ablation (§3.2 of README.md).
# Run this after run_6.sh. It uses the patched training scripts that
# train_ablation_runner.py writes to PATCHED_DIR as a side-effect.
#
# Override any path via environment variables:
#   SCRIPT_DIR    — directory containing eval_cf_ablation.py (default: script dir)
#   DATA_DIR      — directory with fineweb_val_000000.bin (default: ./data)
#   TOKENIZER     — path to bpe_v4096.model (default: ./bpe_v4096.model)
#   CKPT_DIR      — ckpt root from run_6.sh (default: ./ckpt)
#   PATCHED_DIR   — where train_ablation_runner.py wrote patched scripts
#                   (default: /tmp, matches runner default)
#   EVAL_DIR      — output directory for CF eval logs (default: ./eval)
set -e

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
DATA_DIR="${DATA_DIR:-./data}"
TOKENIZER="${TOKENIZER:-./bpe_v4096.model}"
CKPT_DIR="${CKPT_DIR:-./ckpt}"
PATCHED_DIR="${PATCHED_DIR:-/tmp}"
EVAL_DIR="${EVAL_DIR:-./eval}"

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

mkdir -p "$EVAL_DIR"

eval_one () {
  local tag=$1 layers=$2 dim=$3 bdim=$4 xsa=$5
  echo "================================================================"
  echo "== EVAL CF: $tag  (L=$layers d=$dim)"
  echo "================================================================"

  local ckpt_subdir="$CKPT_DIR/${tag}"
  local latest=$(ls -1 "$ckpt_subdir"/step_*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$latest" ]; then
    echo "  NO CHECKPOINT FOUND in $ckpt_subdir"
    return
  fi
  echo "  using checkpoint: $latest"

  # The runner writes /tmp/train_cdm_patched_<L>L_w<weight>.py as a side effect.
  # Pick the one matching this layer count (any weight of the matching scale works
  # for the model class — constants are identical within a scale).
  local patched=$(ls -1 "$PATCHED_DIR"/train_cdm_patched_${layers}L_w*.py 2>/dev/null | head -1)
  if [ -z "$patched" ]; then
    echo "  NO PATCHED SCRIPT for ${layers}L — run run_6.sh first (it writes them). Skipping."
    return
  fi
  echo "  using patched script: $patched"

  python3 "$SCRIPT_DIR/eval_cf_ablation.py" \
    --ckpt "$latest" \
    --train_module_path "$patched" \
    --num_layers $layers --model_dim $dim --vocab_size 4096 \
    --bigram_dim $bdim --xsa_last_n $xsa \
    --n_seqs 500 --seq_len 1024 --stride 2 --rounds 2 --seed 42 \
    --data_dir "$DATA_DIR" --tokenizer_path "$TOKENIZER" \
    --log_path "$EVAL_DIR/${tag}_cf.log" \
    > "$EVAL_DIR/${tag}_eval.out" 2>&1
  echo "  eval done -> $EVAL_DIR/${tag}_cf.log"
  tail -10 "$EVAL_DIR/${tag}_cf.log"
}

# Eval all 6
eval_one 5L_w0    5 256 128 2
eval_one 5L_w03   5 256 128 2
eval_one 5L_w1    5 256 128 2
eval_one 11L_w0   11 512 128 4
eval_one 11L_w03  11 512 128 4
eval_one 11L_w1   11 512 128 4

echo "================================================================"
echo "== ALL 6 CF EVALS DONE"
echo "================================================================"
ls -la "$EVAL_DIR/"
