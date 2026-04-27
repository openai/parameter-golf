# B-WING: Per-Order Entropy Shift + Fixed Order Multipliers

**val_bpb: PENDING** (3-seed mean) | **~15.6 MB** | 8xH100 SXM

## Results

| Seed | val_bpb | Sliding Window BPB | Steps | Train Time | Eval Time | Artifact |
|------|--------:|-------------------:|------:|-----------:|----------:|---------:|
| 1337 | PENDING | PENDING | PENDING | 600s | PENDING | PENDING |
| 42 | PENDING | PENDING | PENDING | 600s | PENDING | PENDING |
| 2024 | PENDING | PENDING | PENDING | 600s | PENDING | PENDING |
| **Mean** | **PENDING** | — | — | — | — | — |
| **Std** | **PENDING** | — | — | — | — | — |

## Approach

X-WING base architecture + three key n-gram eval improvements ported from PR #809:

### 1. Per-Order Entropy Center Shift (from PR #809)

The sigmoid center for alpha computation shifts DOWN for higher n-gram orders:

```
center = entropy_center - 0.25 * (order - min_order)
```

For order 9 (min_order=2): center = 3.0 - 0.25*7 = **1.25**

This means high-order matches fire aggressive alpha even when the model is fairly confident (low entropy). A 9-gram match is so specific that it should override even a confident neural model.

### 2. Fixed Per-Order Multipliers (from PR #809)

Replaces the 3D Cubric adaptive system with proven fixed multipliers:

| Order | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------|---|---|---|---|---|---|---|---|
| Mult | 0.3 | 0.3 | 0.97 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |

Key difference from X-WING: order 4 is **0.97** (near-unity) vs our cubric's **0.45**. PR #809 trusts 4-gram matches much more aggressively.

### 3. Alpha Curve Fix (from PR #809)

| Parameter | X-WING | B-WING |
|-----------|--------|--------|
| alpha_min | 0.20 | **0.05** |
| alpha_max | 0.60 | **0.60** |
| alpha clip | 0.75 | **0.95** |

The 0.95 clip is the biggest lever: with 2.0x multipliers, effective alpha reaches 0.95, letting high-order n-gram matches almost fully override the model.

### Retained from X-WING

- **Complementary training** (COMPLEMENT_ALPHA=0.5): downweight bigram-predictable tokens during training
- **Shared n-gram tables**: all 8 GPU ranks see the full 62M-token picture
- **Score-first protocol**: entire chunk scored before tokens update tables
- **Base architecture**: 11L 512d GQA 8/4, MLP 3.0x, XSA-4, LeakyReLU(0.5)^2, BigramHash(1536), GPTQ int6+zstd
- **8M hash buckets** (vs #809's 4M)

## Eval Stack

- **Backoff cascade**: orders 2-9, 8M flat hash buckets, greedy (highest matching order wins)
- **Entropy-adaptive alpha**: per-order shifted center, `alpha_min=0.05, alpha_max=0.60`
- **Fixed order multipliers**: `(0.3, 0.3, 0.97, 2.0, 2.0, 2.0, 2.0, 2.0)`, clip at 0.95
- **Score-first**: entire chunk scored BEFORE tokens update tables
- **Sliding window**: stride=64, seq_len=2048

## Legality

1. **Score-first protocol**: entire chunk scored BEFORE its tokens update the n-gram tables
2. **Complementary training**: uses only training-data bigram statistics, no validation data during training
3. **Alpha formula**: `(1-a)*P_neural + a*P_ngram` where `a` is a fixed function of model entropy × order multiplier
4. **No oracle selection**: single committed mixture, all tokens have nonzero probability
5. **GPTQ calibration**: runs inside training wallclock

## Timing Budget

| Phase | Time | Notes |
|-------|-----:|-------|
| Training | 600s | ~6800 steps on 8xH100 SXM |
| GPTQ quantization | ~3.4s | Inside training wallclock |
| N-gram eval | PENDING | Shared tables, 8M buckets, orders 2-9 |
| **Total** | **PENDING** | Training + eval |

## Credits

- **Per-order entropy shift + fixed order mults**: @AayushBaniya2006 (PR #809) -- the techniques that close the gap
- **Complementary training**: @travispchen (PR #803)
- **Shared n-gram tables**: @deanbrr (PR #779)
- **N-gram eval cache**: @deanbrr (PR #659)
- **Multi-order backoff + adaptive alpha**: @Asukabot0 (PR #727)
- **X-WING base + 3D Cubric**: @newjordan
- **Base architecture**: @signalrush (PR #414)

## Reproduce

```bash
cd experiments/B_wing/bwing_full_port && SEED=1337 bash run.sh
```

8xH100 SXM, 600s training + eval.
