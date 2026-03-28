# Record: Complementary Training + Backoff N-gram Mixer + Legal TTT

## Summary

- **BPB: 0.4311** (3-seed average: seeds 42, 1337, 2024)
- 11L transformer (26.99M params) with VRL, LeakyReLU(0.5)², XSA-4
- **Complementary training**: model trained with bigram-weighted loss (COMPLEMENT_ALPHA=0.5) to specialize on tokens n-gram caches can't predict
- **BackoffNgramMixer**: orders 2-10, 4M flat hash buckets, greedy cascade (highest order wins)
- **Entropy-adaptive alpha** (0.20 + 0.55*sigmoid(2*(H-3.0))): n-gram gets 20-75% weight based on model uncertainty
- AdamW TTT (lr=5e-4, **3 epochs**, Polyak EMA 0.998, freeze first 9/11 blocks)
- Int6 mixed quantization + lzma compression
- Artifact: max 15,962,841B across seeds (under 16,000,000 byte limit)

## Architecture

- 11 layers, 512 model dim, 8 attention heads, 4 KV heads (GQA)
- 3x MLP expansion with LeakyReLU(0.5)² activation
- BigramHash(2048, dim=128), ValueEmbedding(128, layers 9,10)
- Value Residual Learning (VRL) across all layers
- XSA (Exclusive Self-Attention) on last 4 layers
- U-Net skip connections (encoder-decoder with skip weights)
- SmearGate (learned 1-token look-back)
- Partial RoPE (16 dims), LN Scale

## Key Innovation: Complementary Training

Standard approach: train model on uniform cross-entropy, bolt on n-gram cache at eval.

Our approach: during training, downweight tokens that a bigram predictor would get right (COMPLEMENT_ALPHA=0.5). The model learns to focus its 27M parameters on tokens that statistical caches can't predict — novel word choices, long-range dependencies, semantic surprises.

| Config | BPB |
|--------|-----|
| Base model only | ~1.139 |
| + Standard backoff (alpha=0.05) | ~0.700 |
| + Complementary training + alpha=0.20 | **0.4311** |

## Validated Results (3-Seed)

| Seed | BPB | Artifact (bytes) | TTT Eval Time |
|------|-----|-----------------|---------------|
| 1337 | 0.431107 | 15,916,181 | 477s |
| 42 | 0.431062 | 15,962,841 | 477s |
| 2024 | 0.431112 | 15,958,961 | 475s |
| **Mean** | **0.431094** | **max 15,962,841** | **~476s** |

All runs: training stopped at 600s, full eval (diag+q_rt+q_sw+TTT+ngram) completed in ~562s ≈ 9.37 min.

## Eval Stack

- **BackoffNgramMixer**: orders 2-10, 4M flat hash buckets, greedy cascade
- **Entropy-adaptive alpha**: `0.20 + 0.55 * sigmoid(2*(H - 3.0))`
- **AdamW TTT**: lr=5e-4, 3 epochs/chunk, Polyak EMA 0.998, freeze first 9/11 blocks
- **Sliding window**: stride=64

## Legality

1. **Complementary training**: reweights training loss using training-data bigram statistics only. No validation data accessed during training.
2. **N-gram cache**: built from already-scored tokens only (score-first, backward-looking).
3. **Alpha formula**: fixed function of model entropy, computed before seeing target token.
4. **TTT**: score-first legal TTT on already-evaluated chunks.
5. **Committed distribution**: (1-α)·P_neural + α·P_ngram — proper mixture, all tokens have nonzero probability.

## Reproduction

```bash
# Single seed
VRL_ENABLED=1 LEAKY_RELU=1 TTT_ENABLED=1 TTT_OPTIMIZER=adamw TTT_LR=0.0005 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=2 TTT_TEMPERATURE=0.98 USE_HEDGE_MIXER=1 NGRAM_ORDER=10 NGRAM_BUCKETS=4194304 ALPHA_BASE=0.20 ALPHA_RANGE=0.55 ALPHA_CENTER=3.0 COMPLEMENT_ALPHA=0.5 SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Multi-seed (3-seed validation)
for SEED in 42 1337 2024; do
  VRL_ENABLED=1 LEAKY_RELU=1 TTT_ENABLED=1 TTT_OPTIMIZER=adamw TTT_LR=0.0005 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=2 TTT_TEMPERATURE=0.98 USE_HEDGE_MIXER=1 NGRAM_ORDER=10 NGRAM_BUCKETS=4194304 ALPHA_BASE=0.20 ALPHA_RANGE=0.55 ALPHA_CENTER=3.0 COMPLEMENT_ALPHA=0.5 SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Credits

Based on PR #803 (pentxayc) — Complementary Training + BackoffNgramMixer.
Builds on techniques from: PR #779 (BackoffNgramMixer), PR #549 (LeakyReLU² + TTT), PR #287 (XSA + EMA), PR #413 (VRL), PR #414 (GPTQ-lite base).
