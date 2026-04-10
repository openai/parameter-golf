# Bandit: ClownCar Crawler × Cubric Ngram9

**val_bpb: 0.4961** (3-seed mean, std 0.0003) | **9.21 MB** | 8xH100 SXM

## Results

| Seed | val_bpb | Sliding Window BPB | Post-EMA BPB | Steps | Train Time | Eval Time | Size |
|------|---------:|-----------------:|-------------:|------:|-----------:|----------:|-----:|
| 4 | 0.4964 | 1.1874 | 1.2063 | 7116 | 570s | 168s | 9.27 MB |
| 444 | **0.4957** | 1.1860 | 1.2047 | 7092 | 570s | 168s | 9.21 MB |
| 300 | 0.4961 | 1.1868 | 1.2056 | 7111 | 570s | 168s | 9.52 MB |
| **Mean** | **0.4961** | **1.1867** | — | — | — | — | — |
| **Std** | **0.0003** | — | — | — | — | — | — |

## Architecture

Two components combined:

### 1. ClownCar Crawler Base Model

Frugendorff crawler architecture: 4 flat transformer layers + 1 shared crawler block × 4 loops, `inst_dim=32` FLOW.

- `DELTA_NET_HEADS=0` — causality fix applied (DeltaNet cross-loop state carry removed)
- `EMA_START_STEP=4400`, `EMA_DECAY=0.99`
- `LOOP_AWARE_GPTQ=1` — 2-phase Hessian calibration aware of crawler quantized activations
- `CRAWLER_QUANT_INT8=1`
- Quantized int6+zstd: **~9.2–9.5 MB**

### 2. X-WING Ngram Oracle (from PR #800)

Shared n-gram tables (all 8 ranks update with identical token ranges — full 62M-token picture) + 3D Cubric + complementary training.

- Orders 2–9, 8M hash buckets
- **3D Cubric**: 54 adaptive multipliers (order × entropy_bin × count_bin), warm-start initialized
- **Entropy-adaptive alpha**: 0.20–0.75 via sigmoid on model entropy
- **Complementary training**: `COMPLEMENT_ALPHA=0.5` — downweights bigram-predictable tokens during training
- Score-first: chunk scored before tokens update tables

## Legality

1. **Score-first**: chunk scored before its tokens update ngram tables. No future-looking.
2. **GPTQ timing**: `GPTQ_RESERVE_MS=30000` stops training at ~570s so calibration (~9s) completes within 600s budget. Log confirms: `stopping_early: wallclock_cap train_time:570031ms`.
3. **Complementary training**: bigram table built from training stream only, no val data.
4. **Cubric**: backward-looking beat-rate tracking on already-scored tokens.
5. **Committed distribution**: proper mixture, all tokens nonzero probability.

## Reproduce

```bash
SEED=444 NPROC_PER_NODE=8 bash experiments/Bandit/run.sh
```

8xH100 SXM, ~570s training + ~168s ngram eval.

## Credits

- **ClownCar crawler**: @newjordan (Frugendorff architecture)
- **Causality fix**: DeltaNet cross-loop state carry removed
- **X-WING oracle stack**: @newjordan (PR #800) — shared tables, 3D Cubric, complementary training
- **Shared tables**: @deanbrr (PR #779)
- **Multi-order backoff + adaptive alpha**: @Asukabot0 (PR #727)
- **Complementary training concept**: @travispchen (PR #803)
- **Base architecture**: @signalrush (PR #414)
