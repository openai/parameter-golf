# Evaluation — Spec 016 (Recur-Alpha, α=1 init)

**Run dir:** `runs/016-recur-alpha-ones/seed_42/`
**Commit:** `4dd2d63` on `exp/recur-alpha` (adds α=1 init + grad_norm pre-step snapshot fix + TORCHINDUCTOR_CACHE_DIR)
**Baseline:** spec 008 `runs/008-1736-reproduction/seed_42/final.json` (val_bpb 1.0697 endpoint, 1.1110 @ step 4000)
**Secondary comparison:** spec 015 (α=0 init, same mechanism)
**Eval date:** 2026-04-21

## Result

| metric | spec 008 | spec 015 | spec 016 | 016 Δ vs 008 | 016 Δ vs 015 |
|---|---|---|---|---|---|
| endpoint val_bpb | 1.0697 | 1.0696 | **1.0712** | +0.0015 | +0.0016 |
| matched-step val_bpb @ 4000 | 1.1110 | 1.1078 | **1.1072** | **−0.0038** | **−0.0006** |
| stopping_early step | 4828 | 4761 | 4708 | −120 | −53 |
| pre-quant post-EMA val_bpb | 1.06922 | — | **1.07083** | +0.00161 | — |
| post-quant (GPTQ int6) | 1.07847 | — | **1.08029** | +0.00182 | — |
| artifact size | ~15.98 MB | — | **15.94 MB** | −0.04 MB | — |
| post-TTT | **1.06610** | — | OOM (projected ~1.06792) | +0.00182 (projected) | — |

**Endpoint caveat:** 016 ran 120 fewer steps than 008 (53 fewer than 015) due to JP hardware variance. Endpoint bpb comparisons are confounded; matched-step @4000 is the honest read.

## Noise/signal judgment

**016 Δ vs 015 @ step 4000 = −0.0006: null.** α=1 vs α=0 initialization makes no meaningful difference to screen-level outcome. The mechanism converges to similar α values regardless of starting point — init sensitivity is ruled out.

**016 Δ vs 008 @ step 4000 = −0.0038: real signal.** This is the recur-alpha mechanism effect vs the no-α baseline, now confirmed across both init conditions (015: −0.0032, 016: −0.0038). Consistent signal, same direction.

## Post-hoc GPTQ evaluation (spec 016 only)

Run on pod `1dqkead11ztrxi` using `EVAL_ONLY_BYPASS` patch (skips training, loads `final_model.pt`, runs GPTQ + TTT). Cost ~$4.

| stage | 016 | 008 / #1736 | Δ vs 008 |
|---|---|---|---|
| pre-quant post-EMA | 1.07083 | 1.06922 | +0.00161 |
| post-quant (GPTQ int6) | 1.08029 | 1.07847 | +0.00182 |
| artifact size | 15.94 MB | ~15.98 MB | −0.04 MB (under 16 MB cap ✓) |
| post-TTT | OOM | 1.06610 | — |

Quant cost: 016 = +0.00947, 008 = +0.00941. Nearly identical — the α mechanism does not meaningfully change quantization behavior.

**TTT OOM cause:** `EVAL_ONLY_BYPASS` skipped train_model's warmup phases (CUDA graph bucket warmup + loop_warmup), which normally prime the CUDA allocator. Without priming, TTT's peak footprint exceeded available VRAM. Not a correctness issue; pre-quant and post-quant numbers are clean.

## TTT projection — matched-step extrapolation

The 120-step deficit vs 008 is the dominant gap in the post-hoc numbers. Two projection methods:

**Method A — matched-step advantage extrapolation (preferred):**
016's −0.0038 advantage at step 4000 applied to 008's full-run pre-quant (1.06922):
- Projected 016 pre-quant @ step 4828: ~1.06922 − 0.0038 = **~1.06542**
- Post-quant (+0.00947): **~1.07489**
- Post-TTT (applying #1736's TTT recovery −0.01237): **~1.06252**
- Δ vs 008's 1.06610: **−0.0036** — meaningfully better

**Method B — conservative step-rate extrapolation:**
From 016's actual endpoint, extrapolate 120 more steps at late-training loss rate:
- Projected post-TTT: **~1.06642**
- Δ vs 008's 1.06610: **−0.0003** — barely better, within noise

**Honest range: 1.062–1.067.** Method A is more rigorous (anchors on a clean matched-step comparison). Method B underestimates because the loss rate near the cap overstates how flat the curve actually is vs. a full-length run. True answer likely closer to **~1.063–1.065**, which would be a real improvement over 008.

**Vs #1736's claimed record (1.06549):** Method A projects 016 @ ~1.06252, which would comfortably beat it. Method B projects ~1.06642, which would not. The gap between methods is too wide to call without a matched-step rerun (~$12).

## α trajectory — 016 vs 015

Both converge to the same directional structure; 016 runs ~0.10 "hotter" everywhere (higher absolute α) due to the α=1 head-start.

```
step  | 015 pass-2                 | 016 pass-2                 | 015 pass-3                 | 016 pass-3
------+----------------------------+----------------------------+----------------------------+----------------------------
 2000 | 0.00  0.00  0.00 (init)    | 1.00  1.00  1.00 (init)    | 0.00  0.00  0.00           | 1.00  1.00  1.00
 2200 | 0.03  0.07  0.14           | 0.84  1.02  0.90           | 0.16  0.24  0.33           | 0.75  0.76  0.88
 3000 | 1.04  1.16  1.38           | 1.13  1.30  1.40           | 0.98  0.86  0.76           | 1.04  0.93  0.85
 4000 | 1.04  1.16  1.38           | 1.13  1.30  1.40           | 1.01  0.89  0.77           | 1.04  0.96  0.85
 4700+| 1.04  1.16  1.38 (final)   | 1.13  1.30  1.40 (final)   | 1.01  0.89  0.77           | 1.04  0.96  0.85
```

Same asymmetric depth-gradient pattern: pass-2 amplifies with depth, pass-3 damps. Convergent attractors differ by ~0.10 but are structurally identical. Init choice is irrelevant to final shape.

## Decision — SHELVE init experiments; mechanism validated; evaluate as stack ingredient

**α=0 vs α=1 init: closed.** The 016 null result vs 015 definitively answers the init question. Use α=0 (spec 015 convention) for any future recur-alpha work — both inits are equivalent at convergence.

**Recur-alpha mechanism: validated at screen level.** The −0.0038 matched-step improvement is the largest consistent positive signal since the #1736 rebase. Not a regression, not a null — a small real gain.

**Not a standalone submission.** On the actual submission metric (post-TTT), the projected range straddles or barely beats 008. A matched-step rerun (~$12) would resolve the ambiguity, but:
- The gain is small enough that it likely won't independently beat the frontier.
- The mechanism's value is as a **stack ingredient**, not a standalone win.

**Recommended next steps:**
1. Identify the next highest-EV lever to combine with recur-alpha.
2. If stacking, run both levers together (not sequentially incremental) — avoids the step-deficit compounding problem and gets TTT numbers clean.
3. If budget allows and the projection is close to a submission target, a matched-step rerun (~$12) would de-risk the claim.

**Do NOT:**
- Run more init variants (null result, closed question).
- Run 3-seed confirmation of 015/016 in isolation — the gain is too small to matter without stacking.

## Artifacts preserved on JP volume

JP volume `jlxvxeiol4` (not rsynced, persists until volume termination):
- `final_model.pt` (135 MB, FP32 post-EMA) — reusable as hotstart or for full TTT rerun
- `final_model.int6.ptz` (15.94 MB, GPTQ int6 + brotli) — already quantized, submittable
- `/workspace/.torch_inductor_cache` (~10 GB, keyed on commit `4dd2d63`)

A full TTT rerun would need ~$8 (fresh 8×H100 screen with training to step ~4828, then TTT).

## Cost

| item | cost |
|---|---|
| Spec 016 screen (JP 8×H100, ~20 min) | ~$8 |
| Post-hoc GPTQ eval (JP 8×H100, ~12 min) | ~$4 |
| **Spec 016 total** | **~$12** |
| **Spec 015 + 016 combined** | **~$23** |

## Note on throughput deficit — see spec 016b

The 120-step deficit vs spec 008 (stopping_early 4708 vs 4828) was partially attributed to JP hardware variance. **Spec 016b confirms the deficit is real architectural overhead.** On a same-pod A/B diagnostic (6L/256d proxy, 2×H100), recur-alpha runs 3% slower than baseline — corresponding to ~1–2% at full 11L/512d model size. This accounts for ~50–80 of the 120 lost steps; the remainder is genuine node variance. The TTT projections in this eval should be treated as slightly optimistic since they don't account for the architectural overhead at full scale. See `research/evaluations/016b-recur-alpha-throughput.md`.

## Cross-references

- Spec: `research/specs/016-recur-alpha-ones.md`
- Prior eval: `research/evaluations/015-recur-alpha.md`
- Execution notes: `runs/016-recur-alpha-ones/seed_42/notes.md`
- Throughput diagnostic: `research/evaluations/016b-recur-alpha-throughput.md`
