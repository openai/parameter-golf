# [WIP] Depth Recurrence via Weight-Shared Transformer Blocks

**Status: Validated locally, awaiting GPU throughput test** | Target: < 16 MB | 8xH100 SXM, 600s

## Approach

Weight-shared depth recurrence: instead of 11 unique transformer blocks, share weights across a smaller set of blocks and iterate multiple times, achieving 20+ effective layers within the same 16MB parameter budget.

This technique is listed as an OpenAI "Request for PR" and has not been successfully demonstrated in any prior submission.

## Experimental Results (MLX, local validation)

### Depth recurrence matches baseline quality at 2.8x fewer params

| Config | Params | Effective Depth | val_bpb | Compressed Size |
|--------|--------|-----------------|---------|-----------------|
| Baseline (9 unique layers) | 17.1M | 9 | 3.2273 | 5.1 MB |
| Recurrence 3L x 3R | 6.0M | 9 | 3.2264 | 1.87 MB |

### Deeper recurrence is strictly better

| Config | Params | Effective Depth | val_bpb | Compressed Size |
|--------|--------|-----------------|---------|-----------------|
| Baseline (9 unique layers) | 17.1M | 9 | 3.2273 | 5.1 MB |
| Recurrence 3L x 3R | 6.0M | 9 | 3.2264 | 1.87 MB |
| **Recurrence 3L x 7R** | **6.1M** | **21** | **3.2134** | **1.89 MB** |

The 21-effective-depth model beats the baseline by 0.014 BPB with 2.8x fewer params and 2.7x smaller compressed artifact. Deeper recurrence converges faster at every step count.

### Wider recurrence at full param budget

| Config | Params | Effective Depth | train_loss @ step 20 |
|--------|--------|-----------------|---------------------|
| Recurrence 4L x 5R (768d) | 17.4M | 20 | 5.5597 |

At the same param budget as the baseline, this gets 20 effective layers vs 9 with 1.5x wider hidden dimension. val_bpb pending.

## Parameter Budget Analysis (16MB artifact limit)

| Config | Estimated Artifact | Effective Depth | Notes |
|--------|-------------------|-----------------|-------|
| Current SOTA (11L, 512d) | ~15.9 MB | 11 | Near ceiling |
| 4 unique x 5 reps, 768d | ~13.3 MB | 20 | 1.5x wider, 1.8x deeper |
| 3 unique x 7 reps, 768d | ~10.2 MB | 21 | Most depth-efficient |

## Implementation

Per-iteration conditioning via learned iteration embeddings + sigmoid gating:

```python
# Before each block call at effective layer i:
gate = sigmoid(iter_gate[i])    # starts near 0 (init: -2.0)
x = x + gate * iter_embed[i]   # additive conditioning
x = blocks[i % num_unique](x, x0)  # shared block with cycling
```

U-Net skip connections adapted for effective depth (encoder = first half, decoder = second half with reversed skips).

## Key Risk

MLX shows ~3x throughput penalty per step. Expected 1.5-2x on H100 with torch.compile (weights stay in GPU cache, parameter banks are already stateless). **GPU throughput test is the critical next step.**

## Files

- `train_gpt_mlx_recurrence.py` — MLX prototype (validated, produces all results above)
- `train_gpt_recurrence.py` — CUDA port of baseline with recurrence support (ready for GPU testing)

## Lineage

```
PR #1019 (Current SOTA, 1.1147 BPB)
    +-- This work adds:
        +-- Weight-shared depth recurrence (K blocks x N iterations)
        +-- Per-iteration conditioning (iter_embed + iter_gate)
        +-- Adapted U-Net skip connections for recurrent effective depth
```

## Author

GitGeeks (milhouse)
