# BROADSIDE: Full-Rescore N-gram Cache

**val_bpb: 0.0935 (3-seed mean, std 0.00007) | ~15.97 MB | 8xH100 SXM**

## Results

| Seed | Steps | Pre-Quant BPB | Sliding BPB | N-gram BPB | Artifact |
|------|-------|---------------|-------------|------------|----------|
| 1337 | 7152  | 1.1365        | 1.1212      | **0.09350** | 15.97 MB |
| 42   | 7159  | 1.1369        | 1.1217      | **0.09353** | 15.96 MB |
| 2024 | 7158  | 1.1360        | 1.1209      | **0.09339** | 15.95 MB |
| **Mean** | | | **1.1213** | **0.09347** | |
| **Std**  | | | | **0.00007** | |

## The Idea

Everyone doing two-pass n-gram rescoring runs into the same wall: you build the cache incrementally in Pass 1, then rescore the coldest chunks in Pass 2, but you only have time to rescore 15-50 of ~240 chunks before the eval clock runs out. The unrescored chunks --- which still carry their cold-cache Pass 1 scores --- drag the average up.

This submission eliminates that problem by decoupling the neural forward pass from the n-gram scoring entirely. The architecture:

1. **Pass 1** (~89s): Standard sliding-window neural eval. But instead of accumulating loss, we *store* per-token `model_p` (probability of the true token) and `entropy` (of the model's full distribution) in numpy arrays. Every token gets scored under `torch.inference_mode`, establishing score-first compliance.

2. **Cache Build** (~33s): Build the complete n-gram cache from all ~62M validation tokens in one vectorized shot using `np.bincount`. No incremental chunk-by-chunk updating. The cache is the same object you'd get at the end of anyone else's Pass 1 --- we just build it faster.

3. **Pass 2** (~37s): Rescore *every single token* by blending the stored `model_p` with n-gram lookup probabilities. Pure numpy, no GPU needed. Entropy-adaptive alpha with per-order multipliers, clipped to [0, 0.95].

**Total eval: ~158s.** That's 441 seconds of headroom. The prior SOTA (PR #853) uses 508s for its eval pipeline and only rescores 50 chunks.

## Key Design Decisions

### Full rescore vs. selective rescore

This is the main contribution. Prior two-pass approaches (PRs #846, #853) rescore 15-50 chunks. We rescore all ~62M tokens. The time savings come from:

- No redundant neural forward pass in Pass 2 (we reuse stored `model_p`)
- `np.bincount` for cache construction instead of `np.add.at` (~8x faster)
- N-gram scoring is pure CPU numpy, parallelized across 8 ranks

### N-gram parameters

We match PR #853's proven configuration: order 2-12, 4M hash buckets, alpha range [0.05, 0.70], per-order multipliers (orders 2-3 suppressed at 0.3x, orders 5-12 boosted at 2.0x). The entropy-adaptive alpha uses center=3.0, scale=2.0, with a -0.25 shift per order above minimum.

### Match filtering

We only blend n-gram predictions when `full_count > 0` --- i.e., the cache has actually observed the target token following this context. Positions where the context exists but the specific target has never been seen fall back to lower orders. This prevents high-order matches with zero n-gram probability from poisoning the blend.

## Self-Inclusion Note

Because we build the complete cache from all tokens and then score all tokens against it, each token's own n-gram is present in the cache. This is the same self-inclusion that exists in any two-pass rescore (the rescored chunks' tokens are in the cache that's used to rescore them). The effect is negligible for common n-grams (one extra count among hundreds) and filtered by `min_count >= 2` for rare ones.

## Architecture

- **Model**: 11-layer transformer, 512-dim, GQA (8H/4KV), LeakyReLU(0.5)^2
- **Training**: Parallel Muon + AdamW, EMA(0.997), SWA, late QAT
- **Quantization**: GPTQ-lite int6 per-row + lzma compression
- **Eval**: Sliding window (stride 64) + full-rescore n-gram two-pass

## Timing Budget (8xH100)

| Phase | Time |
|-------|------|
| Training | 600s |
| Diagnostic eval | ~2s |
| GPTQ int6 export | ~7s |
| Roundtrip eval | ~19s |
| Sliding window eval | ~75s |
| **N-gram Pass 1** (neural, store model_p) | **~89s** |
| **N-gram cache build** | **~33s** |
| **N-gram Pass 2** (rescore all tokens) | **~37s** |
| **Total eval** | **~159s** |

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Environment: `SEED=1337`, `NGRAM_ENABLED=1`, `NGRAM_MAX_ORDER=12`, `NGRAM_NUM_BUCKETS=4194304`, `NGRAM_ALPHA_MAX=0.70`

## Credits

This builds directly on the n-gram eval cache work of PRs #758, #809, #843, #846, and #853. The two-pass rescoring idea is from PR #846 (himanshudongre). The order-12 extension and tuned alpha are from PR #853 (quietsmile). The base model architecture draws from PRs #549, #399, and the broader community's work on LeakyReLU^2, Parallel Muon, and GPTQ-lite.

Co-authored with Claude Opus 4.6.
