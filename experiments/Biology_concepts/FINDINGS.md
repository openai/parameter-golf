# Biology Concepts Sweep — Findings Report
**Date:** 2026-03-27
**Run:** `logs/master_20260327_074433`
**Config:** 180s, 8×H100, seed=1337, Green v1 stack

---

## Benchmark Numbers (180s on 8×H100)

| Arm | Steps | Base BPB | Ngram9 BPB | Delta vs Baseline |
|-----|-------|----------|------------|-------------------|
| baseline (green v1) | 2058 | 1.1981 | 0.4742 | — |
| tornado | 1105 | 1.3614 | 0.5221 | +0.048 (worse) |
| theta_gamma | TBD | TBD | TBD | TBD |
| myelin | TBD | TBD | TBD | TBD |
| circadian | TBD | TBD | TBD | TBD |
| astrocyte | TBD | TBD | TBD | TBD |
| clonal_selection | TBD | TBD | TBD | TBD |

**H100 throughput reference:** 87ms/step × 8 GPUs = 2058 steps in 180s

---

## Finding 1: Tornado — EMA Self-Distillation Does Not Help

**Result:** −0.048 BPB (worse than baseline)
**Root cause:** Two compounding problems:

1. **Cold EMA teacher.** The EMA teacher is a running average of all past student states. During early training (first ~500 steps of rapid descent), the EMA is heavily weighted toward initial random weights. The KL signal pushes the student toward a *worse* distribution, not a better one. EMA helps at *convergence* (that's why post-EMA improves val_bpb in normal runs). During the fast descent phase it actively hurts.

2. **Double-forward overhead.** Every KL step requires swapping in EMA weights → teacher forward → swap back → student forward → KL backward. This costs ~400ms per KL step vs ~85ms for normal steps. With cadence=4, average step time = 163ms vs 87ms baseline. Tornado got 1105 steps vs 2058 for baseline — 54% of the training budget.

**KL signal was live and decreasing** (3.31 → 3.16 in first 10 steps), so the mechanism works mechanically. The problem is the *source* of the signal, not the delivery.

**Train loss comparison at equal steps:**
```
step 10:  tornado 5.93 < baseline 6.24  (tornado ahead early — teacher novelty)
step 500: tornado 2.47 > baseline 2.39  (baseline overtook — overhead accumulated)
step 1000: tornado 2.29 > baseline 2.24 (gap widening)
```

---

## Finding 2: Ngram System Rescues Weak Base Models More

**Observation:**
```
baseline: 1.1981 base → 0.4742 ngram9  (rescue: 0.724 BPB)
tornado:  1.3614 base → 0.5221 ngram9  (rescue: 0.839 BPB)
```

A weaker base model gets *more* BPB rescue from the ngram system because it's worse at easy/predictable tokens — exactly the tokens ngram handles. This is a useful calibration: **the base model matters most on hard, non-ngram-predictable tokens.** Any technique that specifically improves the model on hard tokens (low n-gram confidence) will have outsized impact on final combined BPB.

This is the theoretically sound core of tornado's design — focus distillation signal on hard tokens. The problem was the signal source, not the targeting.

---

## Finding 3: What a Real Teacher Would Need

For EMA self-distillation to provide genuine signal, the teacher needs to be *genuinely smarter* than the student. Options:

- **Delay activation:** Don't fire KL until step 1000+ when EMA has converged to a stable, better-than-random state.
- **External checkpoint teacher:** Load a pretrained checkpoint (e.g., from a previous full run) as a frozen teacher. True knowledge distillation.
- **Higher cadence:** cadence=16 or cadence=32 cuts overhead to ~95ms/step average, recovering most of the step budget while still providing occasional teacher signal.

---

## Finding 4: Viable Paths Forward

**If tornado concept worth pursuing:**
- Fix double-forward: cache student logits from CE pass, reuse for KL computation
- Delay tornado activation: `step >= 500` guard
- Test cadence=16 (grid already set up in `experiments/tornado/run_grid.sh`)

**Theoretically cleaner alternatives:**
- `theta_gamma`: Oracle (slow EMA) → Fast Teacher (τ=0.95) → Student. Fast teacher is warm and tracks current student closely. Avoids cold-start problem.
- Hard-token specialist: identify lowest-ngram-confidence tokens, give them higher loss weight directly in CE (no teacher needed).

---

## Infrastructure Notes

- `experiments/Biology_concepts/run_all.sh` — master runner, 180s/arm, auto comparison table
- `experiments/baseline_run.sh` — green v1 config with overridable `MAX_WALLCLOCK_SECONDS`
- `experiments/tornado/run_grid.sh` — 10-arm cadence×KL×temp sweep, ready to run
