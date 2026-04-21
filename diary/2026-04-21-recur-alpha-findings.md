# 2026-04-21 — Recur-Alpha findings (specs 015 + 016)

---

## ⚠️ LATE ADDITION (discovered 2026-04-21 post-017)

Post-017, a latent gap in the recur-alpha patch (commit `a9aa141`) was identified: **the TTT forward path (`_block_with_lora` → called from `forward_ttt` → used by `eval_val_ttt_phased`) does not apply the α blend.** `forward_logits` (training + normal eval) does apply it.

Consequence: every post-TTT number captured on this branch so far (spec 017's 1.06733) was measured on a model where TTT adapted LoRAs and computed loss against an effective α=1 configuration, not against the α-learned one. Training used the α blend; TTT did not.

This doesn't invalidate the training-side findings in this diary (α trajectory shape, depth-monotonicity inversion, path-dependence, late-training rate). Those come from `forward_logits`, which is consistent on-branch.

But it does leave open the question of whether the "partial absorption" story in §6/§7 is partly actual absorption vs partly "TTT was evaluating a different model." Not resolving here — future separate spec. Shelve decision for 017 stands on the observed numbers.

---


Two single-seed screens on learnable per-pass α blending, run earlier today on the #1736 stack. Writing the story up now while fresh; formal evaluations deferred to post-3-seed. A public note may follow.

## 1. What Recur-Alpha is (brief)

In Loop345, layers 3-5 run 3 passes each. Baseline #1736 does unweighted recurrence — every pass fully commits its block output to the residual stream. Recur-Alpha adds a learnable scalar `α` per (extra-pass, looped-layer) position:

```
y = block(x_current)
x_new = α × y + (1 − α) × x_current
```

6 scalars total (2 extra passes × 3 looped layers). At α=0: pure passthrough. At α=1: standard Loop345. At α∈(0,1): partial commitment. At α>1: amplify block, subtract residual.

**Why we ran it.** #1714 tested this on a pre-#1736 stack and got 1.0857 pre-TTT; their compute ran out before phased-TTT composition. Recur-Alpha's behavior on #1736's full stack was literally unmeasured — and looked like the strongest remaining same-parent recurrence lever given how many cousin ideas had already been disproven (#1663, #1726, #1739).

- **Spec 015** (α init = 0): safety-first init. Passthrough at activation, model learns upward.
- **Spec 016** (α init = 1): remove 015's α=0→learned "catch-up handicap" at looping activation.

## 2. Results

Training-endpoint val_bpb (post-EMA, pre-GPTQ, no TTT — screening mode):

| run | α init | endpoint step | endpoint val_bpb | matched @4000 | late-training rate (Δ bpb / Δ step) |
|---|---|---|---|---|---|
| spec 008 | — (baseline) | 4828 | 1.0697 | 1.1110 | 5.0e-5 |
| spec 015 | 0 | 4761 | 1.0696 | 1.1078 | 5.0e-5 |
| spec 016 | 1 | 4708 | 1.0712 | **1.1072** | 5.1e-5 |

### Step-deficit correction

Both α runs ran short vs 008 due to JP pod throughput variance (~1-1.4% slower). The late-training improvement rate is nearly identical across all three runs — ~5.1e-5 per step after step 4000. That means the endpoint gap between 016 and 015 is almost entirely explained by 016 running 53 fewer steps, not by α=1 init being worse.

Extrapolating 016 at the 015 step count (4761):
- 53 more steps × 5.1e-5/step ≈ 0.00270 additional improvement
- Projected 016 endpoint ≈ 1.0712 − 0.0027 = **~1.0685**

If the projection holds, 016 beats 015 by ~0.0011 and beats 008 by ~0.0012 — both ≥5× noise floor (SOTA std ≈ 0.0002). This would make 1.0685 our best training-endpoint number on this stack, period.

**Noise/signal judgment:** matched-step @4000 Δ of −0.0006 (016 vs 015) is in spec 016's "null" bucket, but the step-corrected endpoint picture is consistent with a real −0.001-ish gain. Call it *weak promote with a confound*; 3-seed or matched-clock rerun settles it.

## 3. α trajectories side-by-side

Layout: `[[pass2_L3, pass2_L4, pass2_L5], [pass3_L3, pass3_L4, pass3_L5]]`. Activation fires at step ~2142 (015) / ~2123 (016).

```
step  | 015 α (init=0)                              | 016 α (init=1)
------+---------------------------------------------+---------------------------------------------
 2000 | [[0.00, 0.00, 0.00], [0.00, 0.00, 0.00]]    | [[1.00, 1.00, 1.00], [1.00, 1.00, 1.00]]   init
 2200 | [[0.03, 0.07, 0.14], [0.16, 0.24, 0.33]]    | [[0.84, 1.02, 0.90], [0.75, 0.76, 0.88]]   post-act
 2500 | [[1.00, 1.16, 1.37], [0.85, 0.76, 0.75]]    | —
 3000 | [[1.04, 1.16, 1.38], [0.98, 0.86, 0.76]]    | [[1.13, 1.30, 1.40], [1.04, 0.93, 0.85]]
 4000 | [[1.04, 1.16, 1.38], [1.01, 0.89, 0.77]]    | [[1.13, 1.30, 1.40], [1.04, 0.96, 0.85]]
 4700 | [[1.04, 1.16, 1.38], [1.01, 0.89, 0.77]]    | [[1.13, 1.30, 1.40], [1.04, 0.96, 0.85]]   saturated
```

**Shape is preserved.** Both runs converge to: pass-2 increases with depth (L3 < L4 < L5), pass-3 decreases with depth (L3 > L4 > L5). The *shape* of the plateau reproduces across inits.

**Magnitude is not.** 016's plateau sits ~0.10 higher than 015's everywhere. Same shape, translated.

**Non-monotone exploration in 016.** At step 2200 (just past activation), 016's pass-2 values dipped *below* 1.0 at L3 and L5 (0.84 and 0.90) before climbing back above. α=1 init didn't smoothly drift to its final plateau — it first moved down, then corrected. The loss surface near α=1 has downward gradient on some components before the optimizer finds the upward basin.

## 4. Findings

**Finding 1: α > 1 is preferred on pass-2, especially at depth.** Both 015 and 016 chose values above 1.0 on every pass-2 layer. pass2_L5 saturated at 1.38 (015) and 1.40 (016). The model actively amplifies block output beyond standard residual addition; it's saying "standard Loop345 under-commits, especially at deeper looped layers."

**Finding 2: α < 1 is preferred on pass-3 at deep layers.** pass3_L5 settled at 0.77 (015) and 0.85 (016). Pass-3 under-commits at depth — the third pass through layer 5 contributes less than a standard residual add.

**Finding 3: Depth monotonicity inverts between passes.** Pass-2 climbs with depth (L3 < L4 < L5); pass-3 descends (L3 > L4 > L5). In both runs. This is a specific, non-trivial structure: the two extra passes play *different* roles as a function of layer depth. Simplest mechanistic story: pass-2 overshoots (amplify, inject new direction), pass-3 partially corrects back (damp, stabilize). Hardcoded Loop345 with α=1 everywhere can't express this.

**Finding 4: α plateau is path-dependent.** Different init → different converged α values (same shape, ~0.10 offset). Not a global optimum. α co-evolves with the other model parameters, and the init determines which equivalence-class configuration you land on. This rules out "train once, read off the learned α, then hard-code it" — there is no canonical shape.

**Finding 5: Late-training per-step improvement rate is independent of α plateau.** 008/015/016 all show ~5.0-5.1e-5 val_bpb drop per step after step 4000. The α shape doesn't slow warmdown-phase refinement. Earlier worry (that α>1's negative-weight-on-residual would destabilize late-training) was wrong — refinement proceeds equally well from both plateaus.

## 5. What this tells us about Loop345

- **Fixed α=1 is approximately tuned, not optimally tuned.** Under-commits on pass-2 at depth; over-commits on pass-3 at depth. The baseline ships with the wrong commitment coefficient, and the model tells us so if given 6 degrees of freedom.
- **Two-stage overshoot-correct structure** is what the learned α expresses. A fixed-α architecture cannot do this; making α per-pass-per-layer is the minimum expressiveness needed to recover it.
- **Robustness of shape, fragility of magnitude.** Future recurrence variants should look for constraints that encourage the shape (e.g. structural priors on pass-2 > 1 > pass-3 at depth) without pinning specific values.

## 6. Caveats — what we don't know

- **Single seed per run.** No statistical confidence. Shape could collapse or invert on seed 43/44. Magnitudes almost certainly will shift. Until 3-seed confirms, findings 1-3 are hypotheses, not facts.
- **JP pod variance masked endpoint comparisons.** 016's +0.0016 nominal endpoint regression looked real on first read — it isn't, but catching that required careful per-step rate analysis. A matched-clock rerun or 3-seed would eliminate this confound class entirely.
- **Hardware throughput cost.** 016 got 120 fewer steps than 008. Some fraction of that could be legitimate recur-alpha overhead (tiny extra blend ops × 6 slots × many steps) distinct from pod variance. We haven't isolated this. If real, it applies to every future recur-alpha spec as a hidden tax.
- **TTT composition is untested.** Every number here is pre-TTT, pre-GPTQ. The 1.0685 projection assumes typical TTT/GPTQ gain (~0.003-0.005) on top, giving submission val_bpb ~1.063-1.065. But SpinQuant in specs 009/010 got *fully absorbed* by phased TTT — we can't assume Recur-Alpha survives absorption until we run the full pipeline. This is the single biggest open risk on this whole thread.
- **1.0685 is an extrapolation, not a measurement.** Linear projection at 5.1e-5/step. Late-training improvement might decelerate or accelerate in the final ~50 steps differently. Likely close, but not certain.
- **grad_norm logging in 015 was cosmetic only.** 016's post-fix logs show grad_norm 0.001-0.007 range, confirming autograd was always fine. 015's α values clearly moved (0 → 1.38), which couldn't have happened if grads weren't flowing. Mentioned for completeness — not a data issue.
- **Option B (fixed α at learned shape) is now ruled out.** 015 and 016 converged to *different* α values (same shape, different magnitudes). There's no canonical "learned shape" to freeze. This design is off the table.
- **Pre-training / initial-state sensitivity untested.** Both runs were from-scratch. Behavior when hotstarted from a pre-activation checkpoint is unknown.

## 7. Next steps (conditional)

**Most valuable next move (rank order):**

1. **3-seed 015 (seeds 43, 44) + 3-seed 016 (seeds 43, 44)** in parallel on 4 pods. ~$20. Nails down: (a) whether the −0.001-ish gain is real, (b) whether the α shape reproduces across seeds, (c) resolves the 015-vs-016 init question by seed-averaged comparison.

2. **Full-pipeline run (TTT + GPTQ) on 016 seed 42 resumed from its `final_model.pt`** (which is on JP volume per execution's notes). ~$5-10 on 1×H100 — decouples training from eval, lets us see if Recur-Alpha survives TTT absorption. Should do this *before* burning $20 on 3-seed, because if TTT absorbs the entire gain, the 3-seed exercise is academic.

3. **If TTT composes and 3-seed confirms:** submission-grade full run with 3 seeds at 1.0685-ish training endpoint. ~$60. Produces the number we'd submit.

**Candidates worth considering but not immediate:**
- Cross-pass XSA (orthogonality constraint between pass-2 and pass-3 outputs). Directly tests the overshoot-correct hypothesis. Would use 015's α=0 init (confirmed stable). Can wait.
- α magnitude constraint (clamp α ≤ 1, or reparameterize through sigmoid). Would test whether the α>1 preference is productive or artifactual.

**Ruled out:**
- Option B (fixed α at learned shape) — killed by finding 4.
- α = 0.5 init — redundant given path-dependence; would just find a third plateau.
- Submitting 016 as-is — single-seed + pod variance confound makes it unreliable.

## 8. For a future public note

Short outline for if/when we write this up externally:

> **Headline:** Per-pass, per-depth learnable α blending reveals a pass-2 amplify / pass-3 damp structure in PR #1736's Loop345 recurrence. α > 1 at pass-2, α < 1 at pass-3, depth-monotonicity inverts between passes. Training-endpoint val_bpb improves by ~0.001 vs fixed α=1 baseline, on stack #1736.
>
> **Evidence:** side-by-side α trajectories from two inits (α=0, α=1) — same converged shape, different magnitudes; step-matched val_bpb comparison against baseline; behavior holds on the full #1736 stack (CaseOps + phased TTT + GatedAttn + QuantGate), not a toy subset.
>
> **Caveats:** single seed; pre-TTT numbers; hardware variance partially masks raw endpoint; TTT/GPTQ composition pending.

Not writing the actual note now — waiting on 3-seed + TTT composition first.

## Artifacts

- `runs/015-recur-alpha/seed_42/{train.log,final.json,notes.md}` — local
- `runs/016-recur-alpha-ones/seed_42/{train.log,final.json,notes.md}` — local
- `runs/016-recur-alpha-ones/seed_42/final_model.pt` — on JP volume `jlxvxeiol4`, 135 MB, post-EMA pre-GPTQ. Available as hotstart source for TTT-composition experiment.
- `/workspace/.torch_inductor_cache` on JP volume — populated with 016's compile cache (commit 4dd2d63), cuts next same-commit launch from ~10 min to ~1-2 min.
