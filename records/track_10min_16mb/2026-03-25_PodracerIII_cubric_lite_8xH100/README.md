# Podracing III: Cubric Lite

## Results

| Seed | Sliding BPB | Cubric N-gram BPB | Artifact |
|------|-------------|-------------------|----------|
| 2045 | 1.1193 | **0.9357** | 15.59 MB |
| 43 | 1.1200 | **0.9362** | 15.58 MB |
| 300 | 1.1202 | **0.9365** | 15.58 MB |
| **Mean** | **1.1198** | **0.9362** | — |

## What Changed vs Podracing II (#753)

One eval-time improvement, no training changes:

1. **Per-order adaptive alpha scaling ("Cubric Lite")**: Track how often each n-gram order's probability beats the model's probability on already-scored tokens. Every 32 batches, adjust per-order alpha multipliers. Orders that consistently beat the model get boosted (up to 2.0x), orders that consistently lose get suppressed (down to 0.3x).

**Learned multipliers (converged by step 48):**
```
o2:0.300  o3:0.300  o4:0.970  o5:2.000  o6:2.000  o7:2.000
```

Key insight: bigrams and trigrams (orders 2-3) were actively harming BPB by injecting noisy predictions at the same alpha as high-order matches. Suppressing them to 30% of base alpha and boosting orders 5-7 to 200% = 0.026 BPB improvement over Podracing II (0.9625 → 0.9362).

## Compliance

- Score-first, backward-looking: n-gram cache built from already-scored tokens only
- Alpha depends solely on model's own softmax entropy — no target/label access
- Per-order multipliers use beat-rate statistics from already-scored tokens — same legality as the score-first table update
- No oracle selection, no min-NLL comparison
- GPTQ calibration runs inside training phase (before wallclock stop)
- Cubric multiplier adaptation runs during eval, uses no training data

## Credits

- N-gram eval cache concept: @deanbrr (PR #659)
- Multi-order backoff + adaptive alpha inspiration: @Asukabot0 (PR #727)
- Per-order adaptive alpha scaling (Cubric Lite): @newjordan (original contribution)
- Base architecture: @signalrush (PR #414)

## Reproduce

```bash
SEED=2045 bash concepts/podracer/podracer_green/run.sh
```

8xH100 SXM, 600s training + ~120s eval.
