# Non-Record: Cosine TTT 30 Epochs on SwiGLU + U-Net Architecture (1xH100)

**val_bpb = 1.1175** (sliding window stride=64) | **7.5 MB** artifact | 1xH100 SXM, 600s training + 3376s TTT + 563s eval

## Summary

This submission extends PR #462's SwiGLU + U-Net gated skip architecture with **30-epoch cosine learning rate decay during test-time training** (vs the default 10 epochs with cosine decay). On 1xH100, this single change improves sliding window val_bpb from 1.2531 to 1.1175 (-10.8%).

This finding is consistent with PR #481's independent discovery that cosine TTT scheduling improves results, and PR #486's confirmation that adding 30-epoch cosine TTT improved their stack from 1.1132 to 1.0887 on 8xH100.

## Results (1xH100 SXM, seed 1337)

| Metric | Value |
|--------|-------|
| Training steps | 936 (wallclock capped at 600s) |
| Pre-quant val_bpb | 1.3646 |
| Post-quant roundtrip val_bpb | 1.0684 |
| **Sliding window val_bpb (stride=64)** | **1.1175** |
| Artifact size | 7,505,437 bytes |
| TTT time | 3,376s (30 epochs) |

## Comparison (1xH100, same hardware)

| Config | TTT Epochs | TTT LR Schedule | Sliding BPB |
|--------|:----------:|:---------------:|:-----------:|
| PR #462 defaults | 10 | Cosine | 1.2531 |
| **This submission** | **30** | **Cosine** | **1.1175** |

## Architecture (from PR #462)

- 11 layers, 512 dim, 8 heads, 8 KV heads (full, no GQA)
- Star-ReLU MLP (hidden=1792) with learnable scale+bias
- U-Net skip connections with learned sigmoid gating
- BigramHash (8192 buckets, 128 dim), SmearGate
- EMA (decay=0.9985), Late QAT (threshold=0.15)
- Partial RoPE (16 dims), LN Scale (1/sqrt(layer+1))
- Int6 + zstd-22 compression

## Key Change

```python
# Default (PR #462):
ttt_epochs = 10

# This submission:
ttt_epochs = 30
```

The cosine lr schedule (`ttt_cosine_decay=True`) was already enabled in PR #462. More epochs allow the model to more thoroughly adapt to the validation distribution, with the cosine schedule naturally annealing the learning rate to refine without overshooting.

## Limitation: 8xH100 Timing

On 1xH100, 30 TTT epochs at seq_len=2048 takes ~56 min. On 8xH100, this would be ~7 min (within the 10-min eval budget). However, this needs verification with actual 8xH100 compute. With additional compute credits, we plan to:

1. Verify the 8xH100 timing
2. Tune TTT epochs (20-30) to optimize the quality/time tradeoff
3. Test combined TrigramHash + Value Residual + 30-epoch cosine TTT

## Research Context

Our approach was informed by:
- **Scaling Laws for Precision** (Kumar et al., ICLR 2025): Validated int6 as optimal for our 16MB budget
- **QAT Scaling Laws** (Chen et al., 2025): Informed our quantization timing experiments
- **End-to-End TTT** (Tandon et al., 2025): Motivated exploring TTT scheduling

## What We Tried (Negative Results)

| Experiment | Result | Why |
|-----------|--------|-----|
| Depth recurrence (Huginn-style) | Not competitive | Compute overhead > parameter savings |
| MLP-only TTT (from TTT-E2E paper) | -0.062 BPB worse | Requires meta-learning during training |
| Earlier QAT onset (threshold 0.3) | Slightly worse | QAT slowed convergence |
| Mixed int5/int6 post-training | Catastrophic | Needs int5 QAT during training |

## Credits

- **PR #462** (JoeProAI): SwiGLU + U-Net gated skip architecture
- **PR #481** (mrdavtan): Cosine TTT scheduling discovery
- **PR #442** (sjp611): AdamW TTT
- **PR #398** (felipe-parodi): EMA, TTT, XSA, architectural foundations

## Run Command

```bash
SEED=1337 torchrun --standalone --nproc_per_node=1 train_gpt.py
```

All hyperparameters are set as defaults in train_gpt.py (TTT_EPOCHS=30, TTT_COSINE_DECAY=1).
