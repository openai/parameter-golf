# Beating-1736 note — what it takes, given 016's post-hoc data

**Written:** 2026-04-21 (after spec 016's post-hoc TTT eval OOM'd but captured pre-quant and post-GPTQ numbers)
**Status:** living document; update as runs land

## Target

Beat **#1736's claimed post-TTT val_bpb = 1.06610** on our submission with a meaningful margin (≥0.0005 given ~0.0002 seed std).

## Four-run comparison (all JP 8×H100, 596s wallclock cap)

### Throughput

| metric | 008 | 015 | 016 | #1736 |
|---|---|---|---|---|
| endpoint step | 4828 | 4761 | 4708 | 4854 |
| tok/s at step 4000 | 6.59M | 6.40M | 6.34M | — |
| tok/s at step 4500 | 6.45M | 6.33M | 6.26M | — |
| tok/s at step 4700 | — | 6.29M | 6.21M | — |
| **steps/596s** | **8.10** | **7.99** | **7.90** | **8.14** |
| throughput vs #1736 | 99.5% | 98.2% | 97.1% | 100% |

### Training-endpoint val_bpb

| metric | 008 | 015 | 016 | #1736 |
|---|---|---|---|---|
| endpoint val_bpb (bare) | 1.0697 | 1.0696 | 1.0712 | 1.0696 |
| pre-quant post-EMA | 1.06922 | 1.06916 | 1.07083 | 1.06906 |
| matched-step @4000 | 1.1110 | 1.1078 | 1.1072 | — |

### Post-training pipeline (only 016 and #1736 have these numbers)

| stage | 016 | #1736 | 016 Δ |
|---|---|---|---|
| post-GPTQ (int6) | 1.08029 | 1.07847 | +0.00182 |
| GPTQ cost | +0.00946 | +0.00941 | +0.00005 |
| post-TTT submission | — (OOM) | 1.06610 | — |
| TTT recovery | ? | −0.01237 | ? |

## The chain of assumptions

To go from current 016 measurements to a post-TTT number that beats 1.06610:

```
training endpoint (pre-quant post-EMA)
  → + GPTQ cost (≈ +0.00947)
  → post-GPTQ (int6)
  → + TTT recovery (≈ −0.01237 for #1736)
  → post-TTT submission
```

**Working backward:**

- Target post-TTT: ≤ 1.06610
- If TTT recovery = #1736's −0.01237 → post-GPTQ needs ≤ 1.07847
- If GPTQ cost = observed +0.00947 → pre-quant post-EMA needs ≤ **1.06900**

016's current pre-quant post-EMA at step 4708: **1.07083**.

**Gap to close:** 1.07083 − 1.06900 = **0.00183**.

## How much each lever gives us

At the measured late-training rate of ~5.1e-5 per step:

| scenario | steps gained | bpb improvement | result |
|---|---|---|---|
| Current 016 (step 4708) | 0 | 0 | 1.07083 → miss by 0.00183 |
| +36 steps (step 4744) | 36 | 0.00183 | **just hits 1.06900** |
| Matched to 008 (step 4828) | +120 | 0.00612 | 1.06471 — **beats by 0.00429** |
| Matched to #1736 (step 4854) | +146 | 0.00745 | 1.06338 — **beats by 0.00562** |

**The matched-throughput bar clears the target with 3.3× margin.** The rate analysis says throughput alone is the easy lever.

## Risk ranking — what could kill this

| assumption | current confidence | risk if worse by 0.00183 |
|---|---|---|
| Late-training rate holds at ~5.0e-5/step | High — 3 independent runs show it | Low — would need rate to halve |
| GPTQ cost stays ≤ +0.00947 | High — measured on 016, matches #1736 | Low — already validated |
| **TTT recovery stays ≥ −0.01054** (#1736 got −0.01237) | **UNTESTED** | **HIGH** — recur-alpha × phased-TTT composition is the unknown |

## The one-experiment answer

A single properly-configured run settles the whole story:

- **Matched-clock 016 on NA 8×H100** (kill JP pod variance)
- **Full TTT pipeline** (not the EVAL_ONLY_CHECKPOINT bypass that OOM'd)
- **Bug fix**: the eval-only bypass skipped CUDA-graph + loop warmup, starving the TTT allocator. Fix = restore the warmup phases on the eval path, or just let training-then-eval run in one pod with the training wallclock cap.

Cost: ~$10-15. Delivers:
1. Real post-TTT number, not projection
2. Real NA-throughput baseline to check whether JP variance was the whole story
3. Confirms/disproves "matched-throughput 016 beats #1736 single-seed"

This is the single most valuable next run on this research thread.

## Scenarios by outcome

**Outcome A — Post-TTT ≤ 1.06550** (matches or beats target by ≥0.0005):
- Promote to 3-seed confirmation (~$15-20).
- If 3-seed holds: submission-grade run for leaderboard.

**Outcome B — Post-TTT in [1.06550, 1.06710]** (within ±0.0005 of #1736):
- Too close to call single-seed. 3-seed to resolve.
- Consider stacking with another lever (spec 017 candidate) first.

**Outcome C — Post-TTT > 1.06710** (worse than #1736 by ≥0.0005):
- Recur-alpha gain got absorbed by TTT. Shelf the submission path.
- Keep the α-shape findings as a mechanistic result; don't build further on it.
- Pivot to a non-recurrence lever (cross-pass XSA is a candidate but would need its own TTT composition test).

## What we've learned about #1736-beating regardless of 016

These are observations that survive regardless of 016's outcome:

1. **Matched-step @4000 Δ is the honest comparison for training-endpoint** — raw endpoint comparisons on JP are noisy by ±67-120 steps (~0.003-0.006 bpb). Research should always report matched-step whenever possible.

2. **GPTQ cost is stable at ~+0.00944** across #1736 and 016. Budget this into any projection.

3. **TTT recovery (≈ −0.01237 for #1736) is the "submission multiplier"**, but it's been shown to be ABSORB-capable (SpinQuant in specs 009/010). Don't assume recovery; test it.

4. **Pre-quant post-EMA as a function of step count is nearly linear** across 008/015/016/#1736 (regressed at ~5.0e-5/step, within ±0.0002 of fit). This is useful for quick projections but shouldn't be trusted inside the final ~50 steps where warmdown nonlinearities kick in.

5. **Same-pool JP pod variance is ±1-3% throughput** (~50-150 step deficit on a 596s wallclock). NA pool may be cleaner; haven't measured.

## Not-on-this-thread but related

- If 016 passes the submission bar, next-stacking candidate is a training-time lever orthogonal to recurrence (cross-pass XSA won't stack trivially; training-dynamics levers like tapered WD from spec 011 might).
- Multi-lever stacking math: if we stack a −0.001-bpb training-time lever on top of matched-throughput 016, we move from ~1.06338 to ~1.06238 post-TTT. Cumulative 2-lever path beats #1736 by ~0.004.
