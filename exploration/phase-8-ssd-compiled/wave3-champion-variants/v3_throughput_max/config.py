"""V3 Throughput Max — V1 + larger batch + longer warmdown.
Changes from V1 (clean champion):
  - train_batch_tokens: 524288 -> 786432  (1.5x batch, more GPU utilization)
  - warmdown_iters: 2400 -> 3000          (longer warmdown for more tokens)

Rationale:
  iter-004 hit step 1800 in 10 min at 524K batch. The loss curve was STILL
  steeply decreasing (-0.06 BPB per 100 steps). More tokens = lower loss.

  With 786K batch on 8xH100: each GPU sees 98304 tokens/step (96 seqs of 1024).
  This should be ~1.5x slower per step but with 1.5x more tokens per step,
  giving roughly the same total tokens but with better gradient estimates
  from larger batches.

  If GPU memory allows (iter-004 used 45GB at 524K), 786K should fit in
  the 80GB H100 capacity.

  Longer warmdown (3000 iters) matches leaderboard SOTA practice.
"""

VARIANT_CHANGES = {
    "train_batch_tokens": 786432,
    "warmdown_iters": 3000,
    # All V1 changes inherited:
    "muon_weight_decay": 0.04,
    "grad_clip_norm": 0.3,
    "eval_stride": 64,
    "matrix_lr": 0.03,
    "scalar_lr": 0.03,
    "tied_embed_lr": 0.03,
    "model_dim": 1536,
    "n_iters": 4,
    "d_state": 32,
}

# Run command (8xH100):
# TIE_EMBEDDINGS=0 D_STATE=32 MODEL_DIM=1536 N_ITERS=4 \
# torchrun --standalone --nproc_per_node=8 train_gpt.py

# Run command (1xH100 smoke test, reduced batch):
# TIE_EMBEDDINGS=0 D_STATE=32 MODEL_DIM=1536 N_ITERS=4 \
# TRAIN_BATCH_TOKENS=131072 MAX_WALLCLOCK_SECONDS=300 \
# torchrun --standalone --nproc_per_node=1 train_gpt.py
