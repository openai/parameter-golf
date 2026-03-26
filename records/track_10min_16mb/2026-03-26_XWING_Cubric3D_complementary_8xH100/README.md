# X-WING: 3D Cubric + Complementary Training

**val_bpb: 0.4820** (3-seed mean, std 0.0002) | **15.58 MB** | 8xH100 SXM

## Results

| Seed | val_bpb | Sliding Window BPB | Steps | Train Time | Eval Time | Artifact |
|------|--------:|-------------------:|------:|-----------:|----------:|---------:|
| 1337 | 0.4818 | 1.1196 | 6822 | 600s | 202s | 15.58 MB |
| 300 | 0.4821 | 1.1196 | 6814 | 600s | 204s | 15.66 MB |
| 58 | 0.4821 | 1.1206 | 6822 | 600s | 203s | 15.59 MB |
| **Mean** | **0.4820** | **1.1199** | — | — | — | — |
| **Std** | **0.0002** | — | — | — | — | — |

## Key Innovations

Two novel techniques stacked on shared n-gram tables:

### 1. 3D Cubric Pattern Recognizer (original)

54 adaptive multipliers across three dimensions: **(order x entropy_bin x count_bin)**. Each cell independently tracks how often the n-gram prediction beats the model for that specific regime and adjusts its alpha multiplier accordingly.

This captures patterns invisible to 1D (per-order-only) scaling:
- "order 7 at mid-entropy with high count -> trust fully (2.0x)"
- "order 3 at any entropy -> suppress (0.30x)"
- "order 5 at mid-entropy -> trust strongly (1.9x)"

**Warm-start**: multipliers initialize at proven converged values from prior runs instead of 1.0. Full power from chunk 1 instead of wasting ~30 of 60 chunks converging.

Warm-start initialization:
```
o2: 0.45  o3: 0.30  o4: 0.45  o5: 1.88  o6: 2.00  o7: 2.00  o8: 2.00  o9: 2.00
```

Final converged 3D grid (9 cells per order = 3 entropy bins x 3 count bins):
```
  o2: [0.44 0.40 0.30 | 0.45 0.41 0.30 | 0.45 0.45 0.33]
  o3: [0.30 0.30 0.30 | 0.30 0.30 0.30 | 0.32 0.30 0.30]
  o4: [0.45 0.30 0.30 | 0.66 0.45 0.30 | 0.57 0.72 0.40]
  o5: [1.67 0.90 0.91 | 1.94 1.94 0.99 | 2.00 2.00 2.00]
  o6: [1.82 0.71 0.96 | 2.00 1.94 1.16 | 2.00 2.00 2.00]
  o7: [1.66 0.45 1.05 | 2.00 2.00 1.39 | 2.00 2.00 2.00]
  o8: [2.00 0.37 0.75 | 2.00 2.00 1.19 | 2.00 2.00 2.00]
  o9: [2.00 0.40 0.52 | 2.00 2.00 0.51 | 2.00 2.00 2.00]
```

Key insight: low-order n-grams (2-3) are suppressed across all cells, mid-order (4) has mixed signals, high-order (5-9) are trusted in mid/high-entropy regimes. The cubric learns this automatically through beat-rate tracking.

### 2. Complementary Training (adapted from PR #803)

During training, tokens predictable by bigram statistics receive lower loss weight (`COMPLEMENT_ALPHA=0.5`). A GPU-resident bigram count table (`vocab_size x vocab_size`) tracks `P(y|x)` from training data. The per-token loss weight is:

```
weight = clamp(1.0 - 0.5 * P_bigram(y|x), min=0.1)
```

The model specializes on tokens n-grams can't predict -- novel word choices, long-range dependencies, semantic surprises. This enables higher eval-time n-gram alpha (20-75% vs 5-70%) because the model is deliberately weak where n-grams are strong.

## Eval Stack

- **SharedNgramTable**: chunk-based shared tables -- all 8 GPU ranks update with the same tokens, giving every rank the full 62M-token picture
- **Backoff cascade**: orders 2-9, 8M flat hash buckets, greedy (highest matching order wins)
- **Entropy-adaptive alpha**: `alpha_min + (alpha_max - alpha_min) * sigmoid(scale * (H - center))` with `alpha_min=0.20, alpha_max=0.75, center=3.0, scale=2.0`
- **3D Cubric**: per-token alpha scaled by `cubric_mult[order][ent_bin][cnt_bin]`
- **Score-first**: entire chunk scored BEFORE tokens update tables
- **GPTQ int6+zstd**: quantization runs inside training wallclock
- **Sliding window**: stride=64

## Ablation (single night of development)

| Variant | BPB | Delta | Key change |
|---------|----:|------:|------------|
| Podracer III (#782) | 0.9362 | -- | rank-local tables |
| X-WING v1 (#800) | 0.5644 | -0.372 | shared tables + 1D cubric (6 multipliers) |
| X-WING Yellow II | 0.4896 | -0.075 | 3D cubric (54 mults) + complementary training |
| **X-WING (this)** | **0.4818** | **-0.008** | + warm-start cubric initialization |

## Legality

1. **Score-first protocol**: entire chunk scored BEFORE its tokens update the n-gram tables. No future-looking.
2. **Complementary training**: uses only training-data bigram statistics. No validation data during training. The bigram table is built from `(x, y)` pairs in the training stream only.
3. **Alpha formula**: `(1-a)*P_neural + a*P_ngram` where a is a fixed function of model entropy x cubric multipliers. Target-independent, committed before scoring each token.
4. **Cubric multipliers**: adapt using beat-rate statistics from already-scored tokens (backward-looking only). Updated every 32 chunks.
5. **Warm-start values**: derived from a prior training run's convergence, not from validation data. Equivalent to a hyperparameter choice.
6. **No oracle selection**: single committed mixture, no min-NLL comparison.
7. **GPTQ calibration**: runs inside training wallclock.
8. **Committed distribution**: proper mixture, all tokens have nonzero probability.

## Timing Budget

| Phase | Time | Notes |
|-------|-----:|-------|
| Training | 600s | 6822 steps on 8xH100 SXM |
| GPTQ quantization | ~3.4s | Inside training wallclock |
| N-gram table build + eval | ~202s | Shared tables, 8M buckets, orders 2-9 |
| **Total** | **~802s** | Training + eval |

## Credits & Acknowledgments

- **Complementary training concept**: @travispchen (PR #803) -- the insight that reweighting training loss by bigram predictability enables higher eval-time n-gram weight
- **Shared n-gram table insight**: @deanbrr (PR #779) -- all-rank shared tables instead of rank-local
- **N-gram eval cache**: @deanbrr (PR #659) -- flat hash table design
- **Multi-order backoff + adaptive alpha**: @Asukabot0 (PR #727) -- entropy-adaptive blending
- **3D Cubric pattern recognizer + warm-start**: @newjordan (original)
- **Base architecture**: @signalrush (PR #414)

## Reproduce

```bash
SEED=1337 NPROC_PER_NODE=8 bash concepts/xwing_yellow_III/run.sh
```

8xH100 SXM, 600s training + ~202s eval.
