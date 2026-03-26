# X-WING Yellow II: 3D Cubric + Complementary Training

## Results

| Seed | Sliding BPB | N-gram + 3D Cubric BPB | Artifact |
|------|-------------|------------------------|----------|
| 1337 | 1.1197 | **0.4896** | 15.74 MB |
| **Mean** | — | **0.4896** | — |

Additional seeds pending. Yellow III/IV variants in progress.

## What Changed vs X-WING v1 (#800, 0.5644 BPB)

Three innovations stacked on shared n-gram tables:

1. **3D Cubric pattern recognizer** (original): 54 adaptive multipliers across (order × entropy_bin × count_bin). Each cell independently tracks n-gram beat rates and adjusts its alpha multiplier. The model learns nuanced patterns like "order 7 at mid-entropy with high count → trust fully (2.0x)" vs "order 3 at low-entropy with low count → suppress (0.30x)".

   Converged 3D grid (sample):
   ```
   o7: [0.88 0.30 0.54 | 2.00 2.00 0.65 | 2.00 2.00 1.27]
   o5: [0.97 0.47 0.50 | 2.00 1.70 0.53 | 1.80 1.86 1.38]
   o3: [0.65 0.50 0.47 | 0.72 0.58 0.53 | 0.71 0.69 0.72]
   ```

2. **Complementary training** (adapted from PR #803): During training, tokens predictable by bigram statistics receive lower loss weight (COMPLEMENT_ALPHA=0.5). The model specializes on tokens n-grams can't predict — novel word choices, long-range dependencies, semantic surprises. This enables higher eval-time alpha (20-75% vs 5-70%).

3. **Orders 2-9**: Extended from 2-7. Higher orders contribute meaningfully — cubric gives orders 8-9 multipliers of 1.26-1.30.

## Evolution (single night)

| Variant | BPB | Delta | Key change |
|---------|-----|-------|------------|
| Podracer III (#782) | 0.9362 | — | rank-local tables |
| X-WING v1 (#800) | 0.5644 | -0.372 | shared tables + 1D cubric |
| **X-WING Yellow II** | **0.4896** | **-0.075** | 3D cubric + complementary training |

## Compliance

- Score-first: entire chunk scored BEFORE its tokens update the tables
- Complementary training uses only training-data bigram statistics — no validation data during training
- Alpha is a fixed function of model entropy × cubric multipliers — no target/label access
- Cubric multipliers adapt using beat-rate statistics from already-scored tokens
- No oracle selection, no min-NLL comparison
- GPTQ calibration runs inside training wallclock

## Credits

- Complementary training concept: @travispchen (PR #803)
- Shared n-gram table insight: @deanbrr (PR #779)
- N-gram eval cache: @deanbrr (PR #659)
- Multi-order backoff + adaptive alpha: @Asukabot0 (PR #727)
- 3D Cubric pattern recognizer: @newjordan (original)
- Base architecture: @signalrush (PR #414)

## Reproduce

```bash
SEED=1337 NPROC_PER_NODE=8 bash concepts/xwing_yellow_II/run.sh
```

8xH100 SXM, 600s training + ~182s eval.
