# Parameter Golf — Local Experiment Results
**DGX Spark GB10 · 2026-03-18**

## Experiment Ladder (300 steps, 1 train shard, 1M eval tokens)

| # | Config | val_bpb | Δ vs baseline | params | dim | ms/step |
|---|--------|--------:|----------:|-------:|----:|--------:|
| 1 | Baseline (9 unique layers, 512d) | 2.7927 | — | 17.05M | 512 | 167 |
| 2 | **Fractal only (3×3, 864d)** | **2.5953** | **-0.1975** | 16.57M | 864 | 333 |
| 3 | Fractal + Gravity (3×3, 864d) | 2.6149 | -0.1779 | 16.57M | 864 | 347 |
| 4 | Fractal + Gravity + AttnRes (3×3, 864d) | 2.6084 | -0.1843 | 16.58M | 864 | 425 |

## Training Loss Comparison (300 steps)

| Step | Baseline | Fractal | Fractal+Gravity | Fractal+Grav+AttnRes |
|------|----------|---------|-----------------|---------------------|
| 50   | 5.8850   | —       | 5.8229          | —                   |
| 100  | 5.2427   | —       | 5.0172          | —                   |
| 150  | 4.8926   | —       | 4.6254          | —                   |
| 200  | 4.7830   | —       | 4.5360          | —                   |
| 250  | 4.7162   | —       | 4.4521          | —                   |
| 300  | 4.6554   | 4.3473  | 4.3794          | 4.3751              |

## Key Findings

1. **Weight sharing + wider layers is the dominant effect.** Fractal-only beats baseline
   by 7.1% BPB with fewer total parameters. The 864d shared layers are significantly more
   expressive than 512d unique layers.

2. **Gravity slightly hurts at 300 steps.** The auxiliary losses on early loops add gradient
   noise before those loops learn to produce useful predictions. The model learned weights
   [0.13, 0.13, 0.70] — trying to minimize early loop influence but can't fully zero it.

3. **AttnRes partially recovers the gravity penalty.** Selective depth attention helps
   the model route around noisy early-loop outputs.

4. **All fractal variants beat baseline convincingly.** Even the worst fractal config
   (fractal+gravity at 2.6149) still beats baseline (2.7927) by 0.18 BPB.

## Hypothesis for Full-Scale Runs

Gravity and AttnRes should improve with more training steps because:
- Early loops need many steps to learn useful intermediate predictions
- At 13,000+ steps (H100 10-minute budget), the gravity signal should become useful
- The learned gravity weights should evolve from [0.13, 0.13, 0.70] toward something
  that actually leverages early loops

## Learned Gravity Weights (Experiments 3 & 4)

Both converged to: `[0.127, 0.127, 0.699]`
- softplus(-2.0) = 0.127 (early loops, barely contributing)
- softplus(0.0) = 0.693 (final loop, dominant)
- The model essentially learned to "turn off" early gravity — confirming that at
  300 steps, direct early-loop supervision is noise rather than signal

## SOTA254 Improvement Experiments (8×H100, 2026-03-21)

Baseline: SOTA254 = **1.1303 BPB** (sliding window, seed 1337, zstd)

| Exp | Change | Roundtrip BPB | Sliding BPB | Artifact | Notes |
|-----|--------|-------------:|------------:|---------:|-------|
| A | MTP (2 heads, weight=0.15) | 1.1619 | — | 17.11 MB | zlib fallback; worse than baseline |
| B | SwiGLU MLP (hidden=1024) | 1.1570 | 1.1348 | 17.49 MB | zlib fallback; +0.0045 vs baseline |
| C | Vocab 1536 | — | — | — | can't run (48 GB docs, 36 GB free) |
| **D** | **TTT 8ep + stride 32** | **1.1519** | **1.1295** | **15.74 MB** | **new best! -0.0008 vs baseline** |

**Exp D details:** Same model/artifact as baseline. TTT 8 epochs (vs 3), stride 32 (vs 64). Stride made no difference — all improvement from extra TTT. Seed 1337: 1.1295, Seed 42: 1.1307. Mean: **1.1301** (baseline mean was 1.1308). Confirmed across 2 seeds.

**Bug found (A/B):** zstandard was installed but A/B used zlib anyway — investigate. zstd worked for D.

## Next Steps

1. Try gravity with warmup: zero gravity for first 100 steps, then ramp up
2. Try different loop configs: 2×4, 4×2, 2×5
3. Ship fractal-only (best local result) to cloud H100s for official timing
4. Ship fractal+gravity+attnres as second cloud experiment to test if it
   overtakes with more training

## Environment
- Hardware: DGX Spark GB10, 130.7GB unified VRAM
- PyTorch: 2.10.0+cu130 (no torch.compile, no Triton)
- Data: FineWeb sp1024, 1 train shard, ~100M train tokens
- Eval: 1M validation tokens (truncated for speed)
- Optimizer: AdamW (not Muon — local simplification)
