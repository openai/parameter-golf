# Atris Labs — 10L MLP3x + Int5/Int6 + BigramHash + SmearGate + SWA + Sliding Window

## Approach

Stacked 8 independently validated techniques matching the current leaderboard winners:

### Architecture (25.5M params)
- **10 transformer layers** with U-Net skip connections
- **MLP 3x** expansion (1536 hidden, relu-squared)
- **BigramHash(10240)**: Hash consecutive token pairs into 10240-bucket embedding table (dim=128), zero-init with learnable scale (0.05)
- **SmearGate**: Per-dimension learned gate blending each token with previous token embedding

### Training
- **Muon optimizer**: matrix_lr=0.02, momentum=0.99 (warmup 0.92→0.99 over 1500 steps), weight decay=0.04
- **AdamW**: tied_embed_lr=0.03, scalar_lr=0.02, weight decay=0.01
- **Sequence length**: 2048 tokens, batch 786,432 tokens/step
- **Gradient clipping**: norm=0.3
- **SWA**: Average 24 checkpoints during warmdown (when LR scale < 0.4)
- **Warmdown**: 3000 iterations

### Quantization & Compression
- **Int5 MLP weights** (32 levels, per-row scale) — compresses ~1.88x under zstd
- **Int6 attention weights** (64 levels, per-row scale) — compresses ~1.51x
- **FP16 passthrough** for tied embeddings
- **3% magnitude pruning** before quantization
- **zstd-22** compression (or zlib fallback)

### Evaluation
- **Sliding window eval** (stride=64): every token scored with ~960 context tokens

## Key Metrics

- **val_bpb (int8+zlib roundtrip):** 1.2015
- **val_loss:** 2.0287
- **Artifact size:** 14,386,782 bytes (14.4 MB, under 16MB)
- **Training steps:** 6486 in 600s on 8xH100 (92.5ms/step)
- **Peak memory:** 18,974 MiB
- **SWA:** 24 checkpoints averaged during warmdown

Note: Sliding window eval (stride=64) is implemented but produced worse results
due to a `forward_per_token_loss` interaction with torch.compile/DDP. Standard
eval used for this submission. Fixing sliding window is expected to improve
score by ~0.03 BPB to ~1.17.

## Command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=atris_v8_submission \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=50 \
WARMUP_STEPS=5 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All other hyperparameters use defaults from `train_gpt.py`.
