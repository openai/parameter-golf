# Journal

## Current threads
- Anchor baseline established: exp `0001_baseline_repro` at val_bpb 2.3141, 11.42 MB, 200 steps. All sentinels and noise-floor comparisons reference this.
- MPS stability is built on three knobs in env.sh: `LR_WARMUP_STEPS=10`, `SCALAR_LR=0.01`, `GRAD_CLIP_NORM=1.0`. Don't disable any without a reason logged here.

---

## Entries (newest first)

## 2026-04-24 · exp 0001_baseline_repro · MPS stabilizer stack

**Question**: Can we run a full 200-step smoke on MPS today without NaN?

**Setup**: Default canonical `train_gpt.py` + autoresearch env.sh. Three stabilizers active vs. the historical Apr-18 reference: 10-step linear LR warmup, `SCALAR_LR=0.01` (down from 0.04), `GRAD_CLIP_NORM=1.0`, and `MAX_WALLCLOCK_SECONDS=0` so step-based warmdown actually triggers (the wallclock-based warmdown formula doesn't fire for short smokes).

**Prediction** [LIKELY]: 200 steps clean, val_bpb ~2.3.

**Disconfirming**: any NaN, any step-2 spike >2× step 1, val_bpb outside [2.2, 2.5].

**Result**: val_bpb_post_quant 2.31407318, artifact_mb 11.424, no NaN, no crash, no size violation. step_avg ~1.18 s on MPS. Trajectory: step 1 = 6.94, smooth descent through warmup ramp, hits 4.0 by step 100, ends at train_loss 4.10 with val_loss 3.96.

**Conclusion** [VERIFIED]: The default stabilizer stack works. Three findings worth carrying forward:

1. **Apr-18's reference baseline of 2.5540 was a lucky MPS draw.** Same code, same PyTorch, same seed, same data — today's MPS deterministically NaNs around step 165 with the canonical config, while Apr-18 produced smooth descent. PyTorch MPS is documented as nondeterministic across runs ([pytorch#97236](https://github.com/pytorch/pytorch/issues/97236)). Apr-18 was a single non-reproducible draw. *Do not chase reproducing 2.5540 — chase reproducing 0001_baseline_repro at 2.31.*

2. **The first NaN tracer (`torch.isnan` per param after `opt.step()`) localizes blame to `skip_weights` at step 162.** That's the U-Net skip-connection scaling, optimized by Adam at canonical `SCALAR_LR=0.04`. The default LR is too high for that param's gradient regime — Adam's bias-corrected first-moment denominator can hit a small-`v` underflow, producing a single update large enough to NaN the param. 4× smaller LR keeps the update in range. **`skip_weights` is the first place to look if late-training NaN returns.**

3. **bf16 matmul + Newton-Schulz in Muon was *not* the culprit** despite the natural suspicion. With `MATRIX_LR=0` (Muon frozen), the step-2 spike that occurred at canonical `LR_WARMUP_STEPS=0` was still present — confirming it's the Adam first-step on `tok_emb` (lr=0.05, init std=0.005, ratio 10:1 → catastrophic single update) that drives the early divergence. LR warmup alone fixes that. The late NaN is independent (skip_weights overshoot, see point 2).
