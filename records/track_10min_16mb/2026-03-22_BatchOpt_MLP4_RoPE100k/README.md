# Batch Optimization + MLP4 + RoPE100k

Compared with the baseline 6L-384d run, this version applies a focused set of training and model updates: `TRAIN_BATCH_TOKENS` was reduced from 196,608 to 98,304, `MLP_MULT` was increased from 2 to 4, both `MATRIX_LR` and `SCALAR_LR` were lowered from 0.04 to 0.035, `WARMDOWN_ITERS` was shortened from 800 to 600, and `ROPE_BASE` was raised from 10,000 to 100,000.

In practice, these changes improve optimization efficiency and model capacity while keeping the run inside the 10-minute / 16MB track limits on a single GPU. The best result from this configuration reached **1.4784 val_bpb** on a small GPU (20 GB VRAM) in 10 min.