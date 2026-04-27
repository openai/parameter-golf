# Non-Record Submission: Depth Recurrence + TTT + Gradient Checkpointing

This is a non-record submission exploring **depth-recurrent transformers** with test-time training (TTT) for the Parameter Golf challenge. The architecture uses 3 unique transformer blocks looped 4 times (12 effective layers) with U-Net skip connections, achieving parameter efficiency through weight sharing while maintaining representational depth.

**Status:** Partial results — the 8xH100 benchmark run was interrupted by a RunPod spend limit before final evaluation. We are applying for additional compute credits to complete the 3-run statistical significance requirement.

## Architecture: Depth-Recurrent Transformer

Instead of 12 independent transformer layers, we use **3 unique blocks repeated 4 times** in a U-Net encoder-decoder structure:

- **Encoder half (6 layers):** Store skip connections at each layer
- **Decoder half (6 layers):** Consume skip connections in reverse with learned skip weights
- **Per-iteration differentiation:** Separate RMSNorm parameters and low-rank level signals per effective layer distinguish each loop iteration
- **Weight sharing:** The 3 core blocks (attention + MLP) are shared across all 4 loops

This gives 12 effective layers of depth while only storing 3 blocks worth of unique attention/MLP weights, freeing parameter budget for other components.

### Additional Features

- **BigramHash embeddings:** Hash-based bigram features (10240×128) for local context
- **SmearGate:** Learned gating on the input representation
- **Test-Time Training (TTT):** Optional adaptation on already-evaluated validation tokens during sliding window eval
- **Stochastic Weight Averaging (SWA):** Running average of weights during warmdown phase
- **Sliding Window Evaluation:** Stride-64 evaluation for accurate bpb scoring

## Key Technical Challenges & Solutions

### 1. torch.compile + DDP Incompatibility (NaN)

**Problem:** `torch.compile(fullgraph=True)` combined with DDP produces NaN on the first training step for depth-recurrent architectures. The compiled graph's unrolled loops interact incorrectly with DDP's gradient synchronization hooks.

**Solution:** Use gradient checkpointing instead of torch.compile for DDP training. Each layer's forward pass is wrapped in `torch.utils.checkpoint.checkpoint`, trading ~2x compute for memory efficiency. torch.compile is re-enabled after training for fast evaluation (where DDP is not needed).

### 2. Step-0 Validation Hang

**Problem:** The original code runs validation at step 0 (before any training) because `0 % val_loss_every == 0`. With sliding window eval (stride=64), this creates ~969k forward passes that take 20+ minutes on a single GPU.

**Solution:** Skip validation at step 0 (`step > 0` guard). Use full-stride (fast) evaluation during training, sliding window only for the final evaluation.

### 3. Progressive Loop Training + Warmup Ordering

**Problem:** Progressive loop training (starting with 2 loops, ramping to 4) was configured *after* warmup, causing torch.compile to cache the wrong graph during warmup.

**Solution:** Move progressive loop setup before warmup. Disable progressive loops under DDP (changing loop count creates unused parameters incompatible with DDP).

## Results

### Complete 1xH100 Test (progressive loops enabled, torch.compile)

| Metric | Value |
|--------|-------|
| Steps | 52 |
| Wallclock | 90s |
| val_loss (pre-quant) | 5.7435 |
| val_bpb (pre-quant) | 3.4016 |
| val_bpb (post-quant, int8+zstd) | 3.5041 |
| Peak memory | ~18 GB |
| Step avg | 1,745 ms |

### Partial 8xH100 Benchmark (gradient checkpointing, no compile)

| Metric | Value |
|--------|-------|
| Steps completed | 300+ (interrupted) |
| Wallclock at interruption | ~536s / 600s |
| train_loss at step 300 | 2.4452 |
| Step avg | ~1,786 ms |
| Peak memory per GPU | ~18 GB |

Training was converging well (loss: 26 → 2.45 over 300 steps) when RunPod terminated the pod due to account spend limits. The learning rate warmdown had not yet activated, suggesting further improvement was likely.

### Quick 8xH100 Sanity Test (30s)

| Metric | Value |
|--------|-------|
| Steps | 33 |
| val_bpb | 4.5392 |
| Step avg | 931 ms |

## Configuration

```
NUM_UNIQUE_BLOCKS=3  NUM_LOOPS=4  (12 effective layers)
MODEL_DIM=512  NUM_HEADS=32  NUM_KV_HEADS=4  MLP_MULT=2
VOCAB_SIZE=1024  TRAIN_SEQ_LEN=2048  TRAIN_BATCH_TOKENS=524288
EMBED_LR=0.05  MATRIX_LR=0.04  SCALAR_LR=0.04
SWA_START_FRAC=0.6  EVAL_STRIDE=64
WARMUP_STEPS=50  WARMDOWN_ITERS=1200  MAX_WALLCLOCK_SECONDS=600
```

- Model params: 107,911,265
- DDP: gradient checkpointing (no torch.compile during training)
- Post-training: torch.compile enabled for fast evaluation

## What's Needed to Complete

1. **3 full 8xH100 runs** with different seeds for statistical significance
2. Final post-quantization val_bpb scores
3. Artifact size verification (int8+zstd under 16MB)

We are applying for an OpenAI compute grant to complete these runs.

## Included Files

- `train_gpt_recurrent.py` — Full training script with all fixes
- `submission.json` — Leaderboard metadata
