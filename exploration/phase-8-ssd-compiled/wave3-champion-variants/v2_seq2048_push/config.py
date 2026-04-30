"""V2 Seq2048 Push — V1 + doubled sequence length.
Changes from V1 (clean champion):
  - train_seq_len: 1024 -> 2048  (SSD is O(L), so 2x context != 2x cost)

Rationale:
  All leaderboard top-4 submissions use seq_len=2048.
  SSD has O(L) complexity (vs attention O(L^2)), so doubling context length
  adds ~30-50% step time overhead (chunk count doubles: 16->32 chunks of 64).
  More context per prediction = lower BPB, especially for language modeling.

  The throughput trade-off: ~1200 steps in 10 min (vs ~1800 for 1024).
  But each step carries 2x more context. Net effect should be positive.
"""

VARIANT_CHANGES = {
    "train_seq_len": 2048,
    # All V1 changes inherited:
    "muon_weight_decay": 0.04,
    "grad_clip_norm": 0.3,
    "eval_stride": 64,
    "warmdown_iters": 2400,
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

# Run command (1xH100 smoke test):
# TIE_EMBEDDINGS=0 D_STATE=32 MODEL_DIM=1536 N_ITERS=4 \
# MAX_WALLCLOCK_SECONDS=300 torchrun --standalone --nproc_per_node=1 train_gpt.py
