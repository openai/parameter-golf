# Warmdown Shape Hypotheses

Baseline: WARMDOWN_ITERS=2000 (linear) gave -0.0087 sliding BPB vs 3500 at 200s.
The warmdown shape matters. These hypotheses explore non-linear schedules.
All zero ms/step cost. All testable at 200s.

## H1: Jitter (Stochastic Warmdown)

**Hypothesis:** Gaussian noise in LR during warmdown helps escape local minima.

```
scale = linear_decay * (1 + sigma * randn())
sigma decays 0.3 -> 0 over warmdown
```

- SWA captures diverse checkpoints from noisy trajectory
- Risk: noise near zero LR is meaningless, effective only early warmdown
- Prediction: sliding -0.002 to -0.005

**Config:** `WARMDOWN_MODE=jitter WARMDOWN_JITTER_SIGMA=0.3`

## H2: Swirl (Cosine Oscillation Warmdown)

**Hypothesis:** Cosine oscillations on linear decay create mini warm-restarts inside warmdown.

```
scale = linear_decay * (1 + amp * cos(2pi * cycles * progress))
amp decays with progress
```

- Peaks explore, valleys exploit, SWA averages both
- Each cycle = mini cosine-anneal-with-restart
- Risk: wrong cycle count degrades to jitter or plain cosine
- Prediction: sliding -0.003 to -0.008

**Config:** `WARMDOWN_MODE=swirl WARMDOWN_SWIRL_CYCLES=4 WARMDOWN_SWIRL_AMP=0.3`

## H3: Cascade (Per-Group Warmdown)

**Hypothesis:** Staggered cooldown by parameter group beats uniform schedule.

```
scalar_scale = lr_mul(step, speed=1.5x)   # cool fast
embed_scale  = lr_mul(step, speed=1.0x)   # baseline
bank_scale   = lr_mul(step, speed=0.7x)   # cool slow
```

- Banks carry most capacity, benefit from extended exploration
- Scalars converge early, freezing them reduces gradient noise for banks
- Risk: gradient conflicts if groups are tightly coupled
- Prediction: sliding -0.002 to -0.004

**Config:** `WARMDOWN_MODE=cascade WARMDOWN_SCALAR_MULT=1.5 WARMDOWN_BANK_MULT=0.7`

## Ablation Plan

All tests: 200s wallclock, seed 1337, WARMDOWN_ITERS=2000, siphon/train_gpt.py as engine.
Control = linear warmdown (already have: sliding 1.1760, ngram9 0.4674).

| Test | WARMDOWN_MODE | Key Params | Script |
|------|---------------|------------|--------|
| Control | linear | WD=2000 | rat_rod_ab_test_3_siphon_off (done) |
| H1 | jitter | sigma=0.3 | TBD |
| H2a | swirl | cycles=3, amp=0.3 | TBD |
| H2b | swirl | cycles=5, amp=0.2 | TBD |
| H3 | cascade | scalar=1.5x, bank=0.7x | TBD |

Decision: best warmdown shape gets folded into Siphon full run.

## Priority

1. H2 Swirl — highest ceiling, cleanest SWA interaction
2. H3 Cascade — strongest theory, hardest to tune
3. H1 Jitter — simplest, lowest ceiling, sanity check
