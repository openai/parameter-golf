# Non-record (WIP): Multi-Order N-gram Backoff + Entropy-Adaptive Alpha

**Status: WIP** — validated on 1xH100 SXM proxy run, pending 8xH100 SXM verification for official record.

**Proxy val_bpb = 0.8004** (1xH100, 876 steps, 59% eval coverage) | **15.18 MB** | Seed 42

## Summary

Fork of PR #828 approach (10L + Multi-Order N-gram Backoff) with `MATRIX_LR=0.03`. The n-gram backoff eval cache provides massive BPB improvement over the neural-only model by mixing model predictions with backward-looking n-gram statistics at eval time.

## 1xH100 Proxy Results

| Metric | Value |
|--------|-------|
| Training steps | 876 (1xH100, 600s wall clock) |
| Pre-quant val_bpb | 1.3796 |
| **N-gram eval BPB** | **0.8004** |
| Artifact size | 15.18 MB |
| Eval coverage | 59.4% (570s failsafe) |
| N-gram orders | 2-7, entropy-adaptive alpha |

**Note**: This is a proxy run on 1xH100 with only 876 training steps (vs ~7000 on 8xH100). The base model quality (1.38 BPB) is significantly weaker than what 8xH100 would produce (~1.15 BPB). On 8xH100, we expect the final n-gram BPB to be ~0.90-0.92, consistent with PR #828's reported 0.9076.

## Architecture

- 10L, 512d, GQA 8H/4KV, MLP 3x LeakyReLU(0.5)^2
- BigramHash(4096, dim=128), SmearGate, Value Residual, Gated Attention
- XSA last 4 layers, Partial RoPE 16/64, LN Scale
- U-Net skip connections, tied embeddings, logit softcap=30

## Training

- Muon optimizer: lr=0.03, momentum 0.92 to 0.99, WD=0.04
- EMA(0.997), warmdown=3500 steps
- Mixed int5-MLP/int6-attn quantization + zstd-22
- 3% magnitude pruning

## Eval: Multi-Order N-gram Backoff

- Score-first backward-looking n-gram cache (orders 2-7)
- Highest matching order wins (backoff from 7-gram to bigram)
- Entropy-adaptive alpha: `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))`
- 4M XOR-hash buckets, min_count=2
- **Legal**: each token scored BEFORE cache is updated (Issue #402 compliant)

## Compliance

- [x] Score-first: tokens scored before n-gram cache update
- [x] No pre-eval TTT or adaptation
- [x] No val tokens in artifact
- [x] Artifact under 16 MB (15.18 MB)
- [x] Training under 600s wall clock
- [x] Eval under 570s (failsafe)

## Reproduction

```bash
# 1xH100 proxy (validated):
MATRIX_LR=0.03 SEED=42 torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 official (pending compute access):
MATRIX_LR=0.03 SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Next Steps

- [ ] 8xH100 SXM verification run (3 seeds for statistical significance)
- [ ] Explore frozen n-gram oracle + learned gate (PR #834 approach)
- [ ] Higher-order n-grams (orders 2-9)
- [ ] Complementary training loss weighting

## Based On

- PR #828 (@bigbag): 10L + Multi-Order N-gram Backoff (0.9076 BPB)
- PR #802: Original n-gram backoff implementation
