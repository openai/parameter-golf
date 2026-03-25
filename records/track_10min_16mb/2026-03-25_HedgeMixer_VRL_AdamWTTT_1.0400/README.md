# Record: 1.0400 BPB — Hedge Mixer + VRL + AdamW TTT + Polyak EMA

**3-seed mean: pending (seed 42 = 1.0400, seeds 1337 and 2024 in progress)**

## Summary

11-layer transformer (26.99M params) with Value Residual Learning (VRL), combined with a 5-expert Hedge Mixer during evaluation. The Hedge Mixer blends neural model predictions with online n-gram statistics (unigram, bigram, trigram) using the multiplicative-weights algorithm, achieving a 0.078 BPB improvement over the merged SOTA (1.1194).

## Key Techniques

### Architecture (Training)
- 11 layers, model_dim=512, GQA (8 query / 4 KV heads), MLP 3x expansion
- **Value Residual Learning (VRL):** Residual connection from layer 0's value output to all subsequent layers (~22 extra params, -0.01 BPB)
- **LeakyReLU(0.5)^2** activation
- **XSA-4:** Cross-Token Self-Attention on last 4 layers
- Tied embeddings, BigramHash (2048 buckets), SmearGate
- Soft-Round QAT (late activation at scale < 0.15)
- EMA weight averaging (decay=0.997)
- Muon optimizer (matrix params) + AdamW (embeddings/scalars)
- Int6 mixed quantization + lzma compression

### Evaluation (Legal Score-First)

**TTT (Test-Time Training):**
- AdamW optimizer (lr=0.0005, no weight decay)
- Polyak EMA (decay=0.998): train on raw weights, score with smoothed EMA weights
- Byte-weighted loss: tokens weighted by byte coverage
- Adaptive cosine LR: ramps from 1x to 3x over first 30% of chunks, then cosine decay
- Temperature scaling: T=0.98 on eval logits
- Freeze first 9 blocks, unfreeze last 2 + norms/scales/embeddings (~5.26M of 27M params)
- 3 epochs per chunk, 32K tokens per chunk

**Hedge Mixer (5-expert online ensemble):**
- Expert 0: Neural model (temperature-scaled logits)
- Expert 1: Unigram frequency counts (Laplace-smoothed)
- Expert 2: Bigram P(next|prev) counts
- Expert 3: Trigram P(next|hash(prev2,prev1)) with 64K hashed buckets
- Expert 4: Neural entropy as confidence meta-signal
- Hedge algorithm (eta=0.1) learns expert weights online
- **Deferred weight updates:** Expert weights for chunk N computed from chunks 0..N-1 only. All windows in a chunk scored with identical weights. Updates happen AFTER chunk scoring is complete.
- N-gram tables updated AFTER chunk scoring (score-first, legal)
- Mixer activates after 10,000 tokens accumulated

### Legality

Every token's probability is committed using ONLY information from previously-scored chunks:
1. Hedge weights for chunk N reflect performance on chunks 0..N-1 (deferred update)
2. N-gram tables contain only tokens from chunks 0..N-1 (updated after scoring)
3. Polyak EMA uses fixed decay (0.998), no selection of best snapshot
4. TTT trains only on already-scored chunks (score-first)
5. No validation data accessed during training; no training data accessed during evaluation

## Results

| Metric | Value |
|--------|-------|
| **TTT + Hedge BPB** | **1.0400** |
| Sliding window BPB (no TTT) | 1.1263 |
| Int6 roundtrip BPB | 1.1499 |
| Post-EMA BPB | 1.1425 |
| Training steps | 6104 |
| Step avg | 98.36 ms |
| Artifact size | 15,999,919 bytes |
| Eval time (TTT+Hedge) | 486s / 600s budget |

## Ablation (approximate, from development testing)

| Component | BPB Impact |
|-----------|-----------|
| VRL | -0.010 |
| LeakyReLU(0.5)^2 | -0.001 |
| XSA-4 | -0.005 |
| TTT (SGD baseline) | -0.001 |
| AdamW + Polyak TTT | ~-0.003 vs SGD TTT |
| Hedge Mixer | ~-0.080 |

## Running

```bash
# Full training + eval (single seed):
VRL_ENABLED=1 LEAKY_RELU=1 GATED_ATTENTION=0 TTT_ENABLED=1 \
TTT_OPTIMIZER=adamw TTT_LR=0.0005 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=2 \
TTT_TEMPERATURE=0.98 USE_HEDGE_MIXER=1 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Eval-only (requires saved checkpoint):
EVAL_ONLY=1 CHECKPOINT_PATH=final_model.pt \
VRL_ENABLED=1 LEAKY_RELU=1 GATED_ATTENTION=0 TTT_ENABLED=1 \
TTT_OPTIMIZER=adamw TTT_LR=0.0005 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=2 \
TTT_TEMPERATURE=0.98 USE_HEDGE_MIXER=1 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Acknowledgments

- Hedge Mixer concept inspired by PR #700 (RoyiRa) and PR #702 (lukacf)
- TTT recipe based on PR #606 (EthanYangTW) — AdamW + Polyak EMA + byte-weighted loss
- VRL technique from the Value Residual literature
- Base architecture from PR #198/#287/#549 lineage
