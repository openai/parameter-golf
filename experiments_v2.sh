#!/bin/bash
# experiments_v2.sh — Focused on convergence speed + TTT eval
# Key changes from overnight v1:
# - torch.compile for faster steps
# - Shorter seq (256) for more gradient updates
# - Higher LR sweep (step-limited → need faster convergence)
# - More unique layers / fewer loops (converge faster per step)
# - Standard models that fit 16MB at int6
# - TTT evaluation on best checkpoint

set -e

cd "$(dirname "$0")"
LOGDIR=experiments/logs
RESDIR=experiments/results
CKPTDIR=experiments/checkpoints
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG=$LOGDIR/experiments_v2_$TIMESTAMP.log

mkdir -p $LOGDIR $RESDIR $CKPTDIR

echo "Starting experiments v2 at $(date)" | tee $LOG
echo "==========================================" | tee -a $LOG

# Exp 1: Shorter seq for more steps (same tokens seen but more gradient updates)
echo "" | tee -a $LOG
echo "=== Exp 1: Shorter seq (256) for more steps ===" | tee -a $LOG
python training/train.py --model-type shared \
  --n-unique-layers 6 --n-loops 3 --d-model 512 --n-heads 8 \
  --seq-len 256 --batch-size 128 --train-seconds 600 --compile \
  --exp-name v2_6x3_d512_seq256 2>&1 | tee -a $LOG

# Exp 2: Higher LR for faster convergence in limited steps
echo "" | tee -a $LOG
echo "=== Exp 2: LR sweep ===" | tee -a $LOG
for LR in 1e-3 2e-3 5e-3; do
  echo "--- LR=$LR ---" | tee -a $LOG
  python training/train.py --model-type shared \
    --n-unique-layers 6 --n-loops 3 --d-model 512 --n-heads 8 \
    --lr $LR --train-seconds 300 --compile \
    --exp-name v2_6x3_d512_lr${LR} 2>&1 | tee -a $LOG
done

# Exp 3: More unique layers, fewer loops (converges faster when step-limited)
echo "" | tee -a $LOG
echo "=== Exp 3: Layer/loop configs ===" | tee -a $LOG
for CONFIG in "8 2" "10 2" "12 1" "15 1"; do
  NL=$(echo $CONFIG | cut -d' ' -f1)
  NLOOP=$(echo $CONFIG | cut -d' ' -f2)
  echo "--- ${NL}x${NLOOP} ---" | tee -a $LOG
  python training/train.py --model-type shared \
    --n-unique-layers $NL --n-loops $NLOOP --d-model 512 --n-heads 8 \
    --train-seconds 300 --compile \
    --exp-name v2_${NL}x${NLOOP}_d512 2>&1 | tee -a $LOG
done

# Exp 4: Standard (non-shared) models that fit in 16MB at int6
# int6 = 0.75 bytes/param, 16MB budget → ~22.4M params max
echo "" | tee -a $LOG
echo "=== Exp 4: Standard models ===" | tee -a $LOG
for NL in 4 5 6 7 8; do
  echo "--- standard ${NL}L ---" | tee -a $LOG
  python training/train.py --model-type standard \
    --n-layers $NL --d-model 512 --n-heads 8 --mlp-ratio 3.0 \
    --train-seconds 300 --compile \
    --exp-name v2_standard_${NL}L_d512 2>&1 | tee -a $LOG
done

# Exp 5: Best budget-fitting config from overnight with compile + short seq
echo "" | tee -a $LOG
echo "=== Exp 5: Best overnight config (5x4 d384) with compile ===" | tee -a $LOG
python training/train.py --model-type shared \
  --n-unique-layers 5 --n-loops 4 --d-model 384 --n-heads 8 \
  --seq-len 256 --batch-size 128 --train-seconds 600 --compile \
  --exp-name v2_5x4_d384_seq256 2>&1 | tee -a $LOG

# Exp 6: Evaluate best checkpoint WITH TTT + ngram cache
echo "" | tee -a $LOG
echo "=== Exp 6: TTT evaluation ===" | tee -a $LOG
# Find the best checkpoint from v2 runs by checking CSV files
BEST_CKPT=""
BEST_BPB=99.0
for CKPT in $CKPTDIR/v2_*.pt; do
  if [ -f "$CKPT" ]; then
    BPB=$(python -c "
import torch
c = torch.load('$CKPT', map_location='cpu', weights_only=False)
print(f\"{c.get('val_bpb', 99.0):.6f}\")
" 2>/dev/null)
    echo "  $CKPT: BPB=$BPB" | tee -a $LOG
    if python -c "exit(0 if float('$BPB') < float('$BEST_BPB') else 1)" 2>/dev/null; then
      BEST_BPB=$BPB
      BEST_CKPT=$CKPT
    fi
  fi
done

if [ -n "$BEST_CKPT" ]; then
  echo "Best v2 checkpoint: $BEST_CKPT (BPB=$BEST_BPB)" | tee -a $LOG
  echo "" | tee -a $LOG
  echo "--- TTT: ln_only ---" | tee -a $LOG
  python eval/ttt.py --checkpoint $BEST_CKPT \
    --sgd-steps 4 --lr 5e-3 --adapt-on ln_only \
    --chunk-size 64 --max-eval-bytes 1000000 2>&1 | tee -a $LOG

  echo "" | tee -a $LOG
  echo "--- TTT: all params ---" | tee -a $LOG
  python eval/ttt.py --checkpoint $BEST_CKPT \
    --sgd-steps 4 --lr 1e-3 --adapt-on all \
    --chunk-size 64 --max-eval-bytes 1000000 2>&1 | tee -a $LOG

  echo "" | tee -a $LOG
  echo "--- N-gram cache ---" | tee -a $LOG
  python eval/ngram_cache.py --checkpoint $BEST_CKPT \
    --cache-weight 0.3 --max-order 5 \
    --max-eval-bytes 100000 2>&1 | tee -a $LOG
else
  echo "No v2 checkpoints found, skipping TTT eval" | tee -a $LOG
fi

echo "" | tee -a $LOG
echo "All v2 experiments complete at $(date)" | tee -a $LOG
echo "Log: $LOG"
