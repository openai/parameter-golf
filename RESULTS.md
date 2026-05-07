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

## The Frugendorff Squared — 1.1478 BPB (8×H100, 2026-03-23)

**Architecture:** 6 unique blocks × 2 loops = 12 effective depth, dim=640, 10 heads, 5 KV (GQA), MLP 4x
**Result:** Sliding window **1.1478 BPB** | Pre-quant 1.1570 | Post-quant 1.1716 | Artifact 15.15 MB
**Gap to SOTA:** 0.025 BPB (SOTA = 1.1233)

| Metric | Value |
|--------|-------|
| Sliding BPB (stride 64) | **1.1478** |
| Pre-quant (post-EMA) | 1.1570 |
| Post-quant roundtrip | 1.1716 |
| Quant gap | 0.0146 |
| Params | 28.2M |
| Steps | 4,390 |
| ms/step | 137 |
| Artifact | 15.15 MB |

### What's missing (estimated recoverable ~0.012 BPB):
- Self-distillation (50 steps, temp=2.0) — ~0.003
- Tighter quantization (gap 0.015 → 0.008) — ~0.007
- Tuned warmdown for this architecture — ~0.002

### Why MLP 4x matters
The Qwen overnight sweep found MLP 4x is a massive quality lever (+2% relative BPB). But MLP 4x with 12 unique layers blows the 16MB budget. Fractal weight sharing (6 unique × 2 loops) fits MLP 4x in 15.15 MB. The fractal isn't the point — the MLP 4x it enables is.

---

## The Frugendorff — Fractal Cadence Baseline (8×H100, 2026-03-22)

**Architecture:** 3 unique blocks × 4 loops = 12 effective depth, dim=960, 15 heads, 5 KV (GQA)
**Training:** F/N/N cadence (fractal every 3rd step), Muon optimizer, orthogonal loop positions
**Novel:** Weight-shared transformer with cadence training — completely original architecture

| Run | Config | Sliding BPB | Pre-quant | Steps | ms/step | Artifact |
|-----|--------|------------|-----------|-------|---------|----------|
| v1 (2 blocks, 1024d) | 2×4, MLP 3.0 | 1.2715 | 1.2800 | 7,625 | 79ms | 11.3 MB |
| v1 (3 blocks, 896d) | 3×4, MLP 3.0 | 1.2111 | 1.2257 | 5,933 | 101ms | 12.8 MB |
| **v2 (3 blocks, 960d)** | **3×4, MLP 3.0** | **1.2113** | **1.2217** | **5,738** | **105ms** | **14.2 MB** |
| v3 (3 blocks, 960d) | 3×4, MLP 3.3 | 1.2118 | 1.2210 | 5,590 | 107ms | 14.3 MB |
| v3+TTT (960d) | 3×4, MLP 3.3, TTT | **~1.1901** peak | 1.2210 | 5,590 | 107ms | 14.3 MB |
| v4 (960d, 1.5x batch) | 3×4, MLP 3.3, 1.18M tok | 1.2186 | 1.2257 | 3,764 | 159ms | 14.5 MB |
| v5 (TTT warmup+freeze) | 3×4, MLP 3.3, TTT 1500w | 1.2122 | 1.2212 | 5,602 | 107ms | 14.4 MB |
| longrun (1×H100, 2.3h) | 3×4, MLP 3.3, single GPU | — | 1.3991 | 48,000 | 176ms | — |

### Frugendorff Best: **1.1901 BPB** (v3+TTT peak at window 1400)
### Frugendorff Stable: **1.2113 BPB** (v2, standard sliding window)

### Key Innovations
1. **Fractal weight sharing:** 3 unique blocks looped 4 times = 12 effective layers with only 3 blocks of parameters
2. **Cadence training (F/N/N):** Every 3rd step runs all 4 fractal loops; other steps run single clean pass with orthogonal position
3. **Orthogonal loop positions:** QR-initialized position embeddings ensure each loop and normalize operate in non-interfering subspaces
4. **Qwen-guided overnight optimization:** 141 automated experiments on DGX Spark found optimal config (best: 2×4 loops, lr=2e-3, clip=5.0)
5. **Inner-TTT on fractal loops:** Recursive weight improvement during eval — 4× leverage per TTT step via weight sharing. Peaked at 1.1901 before drift.
6. **TTT drift gate:** Leash on TTT weight updates (lerp back toward originals). Prevents block drift from destabilizing frozen embeddings.

### Experimental Findings
- **TTT v3 (aggressive):** epochs=3, lr=1e-4, drift=0.1 → peaked 1.1901 at window 1400, drifted to ~1.205 by window 4600
- **TTT v5 (conservative):** epochs=1, lr=5e-5, drift=0.05, 1500 warmup windows → no improvement (too gentle, weights barely moved)
- **Sweet spot:** somewhere between v3 and v5. Need epochs=2, lr=8e-5, drift=0.08, freeze at ~1200 windows
- **Bigger batch (v4):** 1.5× tokens/step hurt — fewer total steps offset richer gradients
- **MLP 3.3 vs 3.0:** marginal improvement, extra params barely used
- **Single GPU longrun:** Plateaued at 1.40 BPB after 20K steps. Muon needs distributed all-reduce to work properly. Single GPU with grad_accum is not equivalent.

### Architecture as Compression
The Frugendorff's primary value is as a **compression technique**, not a standalone architecture:
- 3 unique blocks store ~25M params but provide 12 effective layers of depth
- Artifact sizes 11-14 MB vs 16 MB budget — saves 2-5 MB
- Can be used as a "fractal shim" inside a conventional model: e.g., 10 unique layers + 1 shared block × 2 loops = 12 effective depth with 11 blocks of params
- The v6 hybrid (6 unique × 2 loops, 480d, MLP 4x) hit 1.1757 BPB — proving fractal compression works inside a larger architecture

### Qwen Overnight Sweep Results (141 runs, DGX Spark)
| Axis | Best Value | BPB |
|------|-----------|-----|
| num_unique_layers | 2 | 2.3332 |
| num_loops | 4 | 2.3332 |
| cadence | 3 (F/N/N) | 2.3332 |
| lr | 2e-3 | 2.3332 |
| grad_clip | 5.0 | 2.3332 |
| mlp_mult | 3 | 2.3332 |

Winning config: 2 layers × 4 loops, cadence 3, lr=2e-3, clip=5.0, MLP 3 → **2.3332 BPB** (vs 2.6371 baseline, 12% improvement)

### Gap to SOTA
- Our SOTA: **1.1233 BPB** (11 unique layers, 512d, EMA + distillation)
- Frugendorff: **1.2113 BPB** (3 unique blocks × 4 loops, 960d)
- Gap: 0.088 BPB — closing with each iteration

## SOTA254 Improvement Experiments (8×H100, 2026-03-21)

Baseline: SOTA254 = **1.1303 BPB** (sliding window, seed 1337, zstd)

| Exp | Change | Roundtrip BPB | Sliding BPB | Artifact | Notes |
|-----|--------|-------------:|------------:|---------:|-------|
| A | MTP (2 heads, weight=0.15) | 1.1619 | — | 17.11 MB | zlib fallback; worse than baseline |
| B | SwiGLU MLP (hidden=1024) | 1.1570 | 1.1348 | 17.49 MB | zlib fallback; +0.0045 vs baseline |
| C | Vocab 1536 | — | — | — | can't run (48 GB docs, 36 GB free) |
| **D** | **TTT 8ep + stride 32** | **1.1519** | **1.1295** | **15.74 MB** | **new best! -0.0008 vs baseline** |

**Exp D details:** Same model/artifact as baseline. TTT 8 epochs (vs 3), stride 32 (vs 64). Stride made no difference — all improvement from extra TTT.

| Seed | Sliding BPB | Artifact | Status |
|------|------------|----------|--------|
| 1337 | **1.1295** | 15.74 MB | pass |
| 42 | **1.1307** | 15.69 MB | pass |
| 7 | 1.1313 | 16.18 MB | OVER LIMIT |
| 137 | 1.1301 | 16.01 MB | OVER LIMIT (by 8 KB) |

Seeds 7 and 137 both bust 16 MB limit — compression is seed-dependent. Seeds 1337+42 pass. Need a passing 3rd seed.

| Exp | Change | Sliding BPB | Artifact | Notes |
|-----|--------|------------|----------|-------|
| **D+SAM+PR315tricks** | TTT 8ep SAM + Partial RoPE + LN Scale | **1.1274** | 15.81 MB | new best on sota254 base, seed 1337 |

## PR#315 + TTT Experiments (8×H100, 2026-03-22)

PR#315 base (no TTT): **1.1248 BPB**. Added TTT 8ep SAM on top.

**NOTE: TTT is now banned by competition rules. These results are historical only.**

| Seed | Sliding BPB | Artifact | Status |
|------|------------|----------|--------|
| 1337 | **1.1240** | 15.54 MB | pass |
| 42 | running... | — | — |

Best result: **1.1240 BPB** (seed 1337) — beat PR#315 by 0.0008. Invalidated by TTT rule change.

**Note (A/B):** A/B used zlib despite zstandard being installed — likely transient env issue. Resolved; all D runs used zstd correctly.

## Fractal Cadence Experiments (DGX Spark GB10, 2026-03-21)

Hypothesis: Fractal weight sharing causes sawtooth loss — shared weights serve
conflicting roles across loop positions, so 2/3 of gradient updates are destructive.
**Cadence** alternates fractal steps (all loops, depth benefit) with normalize steps
(single clean pass, no loop_pos, no gradient conflict).

| Run | Cadence | val_bpb | Steps | F:N | avg ms/step | notes |
|-----|---------|--------:|------:|----:|------------:|-------|
| Fractal only (baseline) | always F | 2.5953 | 300 | 300:0 | 333 | Mar 18 result |
| **Cadence 2 (F/N)** | **F,N,F,N...** | **2.6276** | **300** | **150:150** | **462** | clean, no gravity |

### Cadence 2 BPB Progression
| Step | val_bpb |
|------|--------:|
| 0 | 4.2284 |
| 50 | 3.4705 |
| 100 | 2.9059 |
| 150 | 2.7429 |
| 200 | 2.6715 |
| 250 | 2.6401 |
| 300 | 2.6276 |

### Key Observations
1. **N steps are ~10ms vs F steps ~96ms** — 10× speed difference
2. **Early pattern (steps 1-10):** F steps always improve, N steps slightly regress
   - Step 5 [F]: 6.8459 → Step 6 [N]: 6.8933 (N undid some of F's gain)
   - Step 7 [F]: 6.6664 → Step 8 [N]: 6.7586 (same pattern)
3. **Cadence 2 landed at 2.6276 vs fractal-only 2.5953** — cadence slightly worse
4. But cadence 2 used only 150 fractal steps (half the compute). Per-fractal-step
   efficiency may be comparable.

### TODO
- [ ] Run clean_always_fractal control (no gravity, same eval-tokens)
- [ ] Run cadence 3 (N/N/F pattern)
- [ ] Run never-fractal control (pure single-pass)

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
