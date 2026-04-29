#!/bin/bash
# V18 Scout: PR #1797 BOS-fixed + tuned hparams from PR #1586/#1787/#1886
# Run with: bash records/track_10min_16mb/2026-04-29_V18_PR1797Tuned_FullStack/run_v18_scout.sh
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-29_V18_PR1797Tuned_FullStack/

SEED=${SEED:-42}
echo "========== V18 SCOUT SEED $SEED [$(date)] =========="

# === V18 hparam stack ===
# PR #1797 dexhunter base: matrix_lr=0.026, attn_clip=13, ttt_lora_alpha=144, warm_start_a=1
# PR #1586 dexhunter GPTQ: MLP_CLIP_SIGMAS=12.0, EMBED_BITS=7, EMBED_CLIP_SIGMAS=15.0
# PR #1787 nprime06 base:  MIN_LR=0.10, GPTQ_RESERVE_SECONDS=0.5
# PR #1886 renqianluo fix: TTT_WEIGHT_DECAY=2.0 (prevent fused CE collapse)

env SEED=$SEED \
  TTT_WEIGHT_DECAY=2.0 \
  MIN_LR=0.10 \
  MLP_CLIP_SIGMAS=12.0 \
  ATTN_CLIP_SIGMAS=13.0 \
  EMBED_BITS=7 \
  EMBED_CLIP_SIGMAS=15.0 \
  GPTQ_RESERVE_SECONDS=0.5 \
  TTT_LORA_ALPHA=144 \
  TTT_WARM_START_A=1 \
  MATRIX_LR=0.026 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/scout_v18_seed${SEED}.log 2>&1

echo "========== V18 SCOUT DONE [$(date)] =========="
echo "=== Final BPB ==="
grep -E "post_ttt_val_bpb|sliding_val_bpb|val_bpb|Total submission|stopping_early" /workspace/scout_v18_seed${SEED}.log | tail -25
