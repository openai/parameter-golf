# Record: Order-Adaptive 9-gram Backoff + Distributed Prefill — val_bpb 0.4405 (3-seed mean)

## Results

| Seed | val_bpb | Artifact | Eval time |
|------|---------|----------|-----------|
| 42 | 0.4429 | 14,899,126 bytes | ~586s |
| 1337 | 0.4381 | 14,740,261 bytes | ~588s |
| 2024 | 0.4405 | 15,101,371 bytes | ~502s |
| **Mean** | **0.4405** | | |
| **Std** | **0.0024** | | |

- Artifact: < 16,000,000 bytes (all seeds)
- Train: 600s on 8xH100 SXM
- Eval: < 600s (all seeds)

## Method

11-layer transformer (512d, 8/8 full MHA, XSA-all, LeakyReLU(0.5)², 3.5x MLP).
Order-adaptive entropy-gated 9-gram backoff cache with per-order entropy thresholds
and distributed cache prefill. Score-first, backward-looking, deterministic.

### Architecture
- 11L, 512d, full MHA 8/8, MLP 3.5x (1792), LeakyReLU(0.5)²
- XSA on all 11 layers, partial RoPE 16/64
- BigramHash(4096, 128d), SmearGate, VE128 on layers 9-10
- Tied embeddings, logit softcap 30
- EMA(0.997) + Tight SWA, Parallel Muon optimizer
- int5 per-row quantization + zstd-22 compression
- Early QAT (threshold 0.5)

### Eval-time N-gram Cache
- Multi-order backoff, orders 2-9, 4M hash buckets per order
- Dual hash tables per order: context counts + full (context+target) counts
- Per-order entropy thresholds: {9: 2.6, 8: 2.8, 7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5}
- Entropy-adaptive alpha: 0.05 + 0.55 * sigmoid(2.0 * (H - threshold))
- Alpha range [0.05, 0.60]: low entropy = trust neural, high entropy = trust n-gram
- min_count=2, score-first (lookup then update per window)
- Distributed prefill: each rank pre-warms cache with all preceding token positions
- Sliding window eval with stride=32

### Key Insight
Distributed cache prefill is critical — without it, ranks 1-7 start with cold caches,
losing ~60% of n-gram effectiveness. Prefill makes distributed eval equivalent to
single-GPU sequential eval. Combined with 9-gram orders (capturing longer repeated
phrases) and per-order entropy gating (trusting higher orders at lower uncertainty),
this produces a -0.69 BPB gain over neural-only sliding window eval.

## Legality

- **Score-first n-gram cache**: Each window batch: (1) lookup cache for predictions,
  (2) compute blended loss, (3) update cache with window tokens. Cache only uses
  backward-looking tokens that have already been scored. No future data access.
- **Alpha depends on model entropy only**: The mixing weight uses the neural model's
  output entropy, not the target token. No oracle/hindsight selection.
- **No TTT**: Test-time training is disabled (TTT_EPOCHS=0).
- **No GPTQ at eval time**: Quantization completes within the training budget.
- **No reordering**: Evaluation set processed in original sequential order.
- **Deterministic**: Given the same seed, produces identical results.

## Acknowledgments

Huge thanks to the incredible community:

- @abaybektursun (PR #549) — base architecture + Legal TTT + Parallel Muon
- @deanbrr (PR #659, #779) — invented the n-gram eval cache, BackoffNgramMixer
- @Asukabot0 (PR #715, #727) — entropy-adaptive alpha formula
- @Robby955 (PR #796) — distributed cache prefill technique
- @hypery11 (PR #788, #795, #825) — order-adaptive entropy gating, 9-gram extension
- @newjordan (PR #753, #782) — multi-order backoff, per-order alpha scaling
- @travispchen (PR #798) — per-order entropy thresholds
- @gowtham0992 (PR #606) — int5 + QAT
- @signalrush (PR #414) — EMA training recipe
- @thwu1 (PR #180) — mixed quantization, BigramHash, SmearGate
- @raahilshah (PR #162) — int6 quantization foundation
