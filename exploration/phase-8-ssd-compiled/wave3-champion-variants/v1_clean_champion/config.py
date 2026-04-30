"""V1 Clean Champion — iter-004 core + proven v7 improvements.
Changes from iter-004 baseline (1.3196 BPB):
  - muon_weight_decay: 0.0 -> 0.04  (leaderboard-proven regularization)
  - grad_clip_norm: 1.0 -> 0.3      (tighter clipping, proven in leaderboard top-4)
  - eval_stride: 0 -> 64            (sliding window eval, ~0.034 BPB free)
  - warmdown_iters: 1200 -> 2400    (longer LR decay, matches leaderboard)
  - matrix_lr: 0.02 -> 0.03         (matches iter-004's actual env-var override)
  - scalar_lr: 0.02 -> 0.03         (matches iter-004's actual env-var override)
  - tied_embed_lr: 0.02 -> 0.03     (matches iter-004's actual env-var override)

NOT changed (preserves iter-004's working core):
  - No vertical state carry (x, _ = block(...))
  - No Triton kernel
  - No bifurcated A_log (log-uniform [-4.5, 0.5])
  - No find_unused_parameters
  - d_state=32, d=1536, n_iters=4, expand=2
"""

VARIANT_CHANGES = {
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
# TIE_EMBEDDINGS=0 D_STATE=32 MODEL_DIM=1536 N_ITERS=4 MATRIX_LR=0.03 SCALAR_LR=0.03 \
# torchrun --standalone --nproc_per_node=8 train_gpt.py

# Run command (1xH100 smoke test):
# TIE_EMBEDDINGS=0 D_STATE=32 MODEL_DIM=1536 N_ITERS=4 MATRIX_LR=0.03 SCALAR_LR=0.03 \
# MAX_WALLCLOCK_SECONDS=300 torchrun --standalone --nproc_per_node=1 train_gpt.py
