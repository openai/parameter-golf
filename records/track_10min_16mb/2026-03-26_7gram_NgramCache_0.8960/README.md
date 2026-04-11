# 7-gram N-gram Cache + XSA-All + LeakyReLU² + Parallel Muon

**val_bpb: 0.8960** (3-seed mean, std 0.0004) | **~15.92 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-cache bpb | **Post-cache bpb** | Cache gain | Eval time | Artifact |
|------|----------|-------|---------------|-------------------|------------|-----------|----------|
| 1337 | 102.5ms | 5,855 | 1.1488 | **0.8956** | -0.2532 | 136s | 15,917,799 |
| 42 | 102.6ms | 5,847 | 1.1488 | **0.8959** | -0.2529 | 136s | 15,921,427 |
| 2025 | 102.7ms | 5,846 | 1.1500 | **0.8966** | -0.2534 | 138s | 15,911,611 |
| **Mean** | **102.6ms** | **5,849** | **1.1492** | **0.8960 (std 0.0004)** | **-0.2532** | **~137s** | |

## Key Innovation: Streaming 7-gram N-gram Cache at Eval

The dominant technique. A single-pass streaming n-gram cache built during sliding window evaluation delivers **-0.2532 bpb** over the base model -- without TTT, without an online mixer, without any weight updates at eval.

### How it works

During sliding window evaluation (stride 64), we maintain hash-based frequency tables for n-grams of order 2 through 7. For each scored token:

1. **Score**: Compute neural model NLL via forward pass under `torch.inference_mode()`
2. **Cache lookup**: For each n-gram order (2-7), hash the preceding context tokens to look up how often this exact n-gram has been seen before. Higher-order matches override lower-order via near-zero backoff:

```python
# Recursive backoff: higher-order n-gram overrides lower with near-zero beta
cache_prob = (min(full_count, ctx_count) + beta * prev_cache_prob) / (ctx_count + beta)
# beta = 1e-6: effectively, if a 7-gram match exists, use it; else fall back to 6-gram, etc.
```

3. **Blend**: Fixed 80/20 mix of cache and neural probabilities on tokens where the cache has >= 2 observations:

```python
blend_prob = 0.80 * cache_prob + 0.20 * neural_prob  # fixed alpha = 0.20
```

4. **Update**: After scoring, add the token's n-grams to the frequency tables (score-before-update)

### Why it works so well

Web text (FineWeb) is highly repetitive at the n-gram level. Common phrases, boilerplate, HTML patterns, and formulaic language recur frequently across documents. A 7-gram cache captures these patterns with high specificity -- a 7-token match is almost always a strong predictor of the next token. The near-zero backoff beta (1e-6) ensures that when a long match exists, it dominates completely, falling back gracefully to shorter contexts only when needed.

The fixed alpha of 0.20 (80% cache, 20% neural) was found empirically to outperform both entropy-adaptive blending and the Hedge online mixer. The neural model's contribution is modest but consistent on tokens the cache hasn't seen before.

### Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| N-gram order | 7 | Orders 2-7, hash table per order |
| Hash table size | 2^22 (4M entries) | Per order, uint32 counts |
| Backoff beta | 1e-6 | Near-zero: longest match dominates |
| Alpha (blend weight) | 0.20 fixed | 80% cache, 20% neural |
| TTT | Disabled | Cache alone outperforms TTT |
| Online mixer | Disabled | Fixed blend outperforms Hedge |

### What we tried and discarded

| Technique | Result | Why discarded |
|-----------|--------|---------------|
| Hedge mixer (5 experts) | 1.0166 bpb | Mixer converges to blend=1.0, doing nothing useful |
| TTT (SGD, 3ep) | 1.0308 bpb | Adds ~410s eval time, cache alone is better |
| PPM exclusion | 0.9909 bpb | Marginal gain, adds complexity |
| PAQ-style log-odds mixing | ~0.99 bpb | No gain over linear blend at this alpha |
| 5-gram cache (backed off) | 0.9884 bpb | 7-gram is strictly better |
| Phase-2 cache replay | 0.8183 bpb | **INVALID**: information leakage, same as PR #611 |

## Legality

- **Single-pass, score-before-update**: Each token is scored using only cache statistics from previously scored tokens. Cache update happens after scoring (lines 1206-1215 run after loss accumulation at line 1194).
- **No TTT**: `TTT_ENABLED=0`. No weight updates during evaluation.
- **No training on validation data**: Training loop uses `fineweb_train_*.bin` exclusively.
- **No multi-pass replay**: Each `eval_val_sliding` call initializes fresh caches. No re-scoring.
- **`torch.inference_mode()`**: All eval runs under inference mode, preventing any gradient computation or weight mutation.

## Training Architecture

Built on PR #549 stack (LeakyReLU² + Legal TTT + Parallel Muon):

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 3x with LeakyReLU(0.5)² |
| XSA | **All 11 layers** (extended from last-4) |
| BigramHash | 2048 |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parallel Muon |
| SmearGate | Enabled |
| U-Net skips | Enabled |
| Tied FP16 embeddings | Softcap 30.0 |

## Timing Budget

| Phase | Time |
|-------|------|
| Training | 600s (=10 min wallclock cap) |
| Standard eval (post-EMA + int6 roundtrip) | ~9s |
| **Sliding window eval + 7-gram cache** | **~137s** |
| **Total eval** | **~146s (< 3 min)** |

Total wall time: ~746s (~12.4 min). Well within 10 min train + 10 min eval budget.

## Run Command

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All n-gram cache parameters are baked into the defaults in `train_gpt.py`. Key env vars and their defaults:

```
NGRAM_ENABLED=1  NGRAM_ORDER=7  NGRAM_ALPHA_LOW=0.20  NGRAM_ALPHA_HIGH=0.20
NGRAM_BACKOFF_BETA=1e-6  TTT_ENABLED=0  USE_MIXER=0
```

## Ablation

Incremental contribution of each technique (seed 1337, same trained model):

| Change | bpb | Delta |
|--------|-----|-------|
| PR #549 base (int6 roundtrip, no sliding window) | 1.1488 | -- |
| + Sliding window eval (stride 64) | 1.1422 | -0.0066 |
| + 2-5gram cache (backed-off, entropy-adaptive alpha) | 0.9884 | -0.1538 |
| + Extend to 7-gram | 0.8972 | -0.0912 |
| + Near-zero backoff beta (1e-6) + fixed alpha 0.20 | **0.8956** | -0.0016 |

## Credits

- **Base model (LeakyReLU² + Legal TTT + Parallel Muon)**: [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun
- **XSA-all extension**: [PR #745](https://github.com/openai/parameter-golf/pull/745), [PR #740](https://github.com/openai/parameter-golf/pull/740)
- **N-gram cache inspiration**: [PR #741](https://github.com/openai/parameter-golf/pull/741) (Cosine TTT + Multi-Order N-gram Cache, 0.9850 bpb)
- **Hedge mixer**: [PR #745](https://github.com/openai/parameter-golf/pull/745) by multiple contributors
- **Backoff/blending techniques**: Inspired by PAQ/cmix compression literature (Matt Mahoney, Byron Knoll)
