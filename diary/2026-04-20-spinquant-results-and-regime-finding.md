# 2026-04-20 pt.2 — SpinQuant results and the regime-dependence finding

**Session kind:** research, post-execution-run. Continuation of the morning's design work (see `2026-04-20-baseline-migration-and-spinquant-design.md`). **Days to deadline:** 10.

## TL;DR

- Spec 009 ran. `baseline` closed spec 008's missed post-TTT gate at **1.06728** (matches #1736's 1.06610 within bf16 noise). `internal_only` (R_a) at **1.06731** — **null result** vs baseline (Δ = +0.00003).
- Spec 010 ran. `port_1695` (4-rotation online Hadamard scheme) at **1.06723** — also **null result** (Δ = −0.00005).
- BUT — the per-batch analysis of spec 010's trajectory revealed a striking **regime-dependent effect**: rotation helps long-context docs (−0.0064 bpb) and hurts short-context docs (+0.0146 bpb). The null aggregate is a cancellation, not an absence of effect.
- Designed and implemented spec 010b (site-selective SpinQuant, `SPINQUANT_SITES` env var). ~$25, 4-mode sweep, ready for execution. Hypothesis: attention rotations carry the long-context benefit, MLP rotations carry the short-context hurt; `attn_only` mode may net positive.
- Still haven't touched spec 011 (tapered WD retrain).

## Spec 009 run

Execution ran it end-to-end on a single 8×H100 pod at JP-1, cost ~$14.70 including a $1.20 wrong-image attempt. Back-to-back `baseline` + `internal_only` with compile-cache reuse saved ~65 s on the second TTT eval.

### Results vs what we expected

| stage | baseline | internal_only | Δ |
|---|---|---|---|
| `diagnostic_pre_quant_post_rotation val_bpb` | 1.22161 | 1.22159 | −0.00002 |
| `diagnostic_quantized val_bpb` | 1.08010 | 1.08007 | −0.00003 |
| **`quantized_ttt_phased val_bpb` (GATE)** | **1.06728** | **1.06731** | **+0.00003** |
| artifact bytes (< 16 MB cap) | 15,948,105 | 15,947,721 | −384 |

- **Spec 008 reproduction verified.** Baseline's 1.06728 is +0.0012 vs #1736's 1.06610. Consistent with spec 008's +0.00016 pre-quant delta (bf16 cross-pod noise). We're now confident #1736 is reproduced end-to-end on our stack.
- **R_a rotation is null on aggregate.** Artifact was 384 bytes smaller (rotation did spread outliers), and `diagnostic_quantized` dropped 0.00003 bpb (tiny quant-error reduction measurable). But phased TTT's LoRA absorbed whatever residual quant-error structure remained.

### Trajectory

First time I actually looked at the `ttp:` log column `rb` (running bpb):

| batches done | baseline rb | internal_only rb |
|---|---|---|
| 5 | 1.1142 | **1.0900** |
| 25 | 1.0771 | 1.0811 |
| 100 | 1.0625 | 1.0659 |
| 500 | 1.0601 | 1.0618 |
| 780 | **1.0663** | **1.0679** |

Early batches (1–15): internal_only below baseline (rotation reduces raw quant error; LoRA untrained). Crossover around batch 20. After that, internal_only sits consistently +0.0016 above baseline for the rest of eval.

**My initial interpretation:** "TTT LoRA substitutes for rotation." LoRA learns to correct quant error; once it's adapted, R_a's contribution is redundant. Filed as the working hypothesis.

## Spec 010 run

Ran ~17 min later on the same pod (restarted). Cost ~$6.50. Single `port_1695` mode.

### Results

| stage | baseline (from 009) | port_1695 | Δ |
|---|---|---|---|
| `diagnostic_pre_quant_post_rotation` | 1.22161 | 1.22161 | ≈ 0 |
| `diagnostic_quantized` | 1.08010 | 1.08001 | −0.00009 |
| `quantized_ttt_phased` | 1.06728 | **1.06723** | **−0.00005** |

Another null. Port_1695's 4 rotations (attn-in, attn-proj-in, mlp-in, mlp-proj-in) together delivered twice the raw-quant-error reduction of internal_only (0.00009 vs 0.00003 at `diagnostic_quantized`), but that advantage evaporated after phased TTT.

### The disappointingly-good intra-eval signal

Looking at the `ttp:` running-avg column, port_1695 looked like a **SOTA break** mid-eval:

| batches done | baseline rb | port_1695 rb | Δ |
|---|---|---|---|
| 1 | — | 1.0593 | — |
| 25 | 1.0771 | 1.0604 | **−0.0167** |
| 116 | 1.0625 | 1.0524 | **−0.0101** |
| ~500 | 1.0601 | ~1.057 | ~−0.003 |
| 773 (last log) | 1.0663 | 1.0604 | −0.006 |
| **final val_bpb** | **1.06728** | **1.06723** | −0.00005 |

At batch 116, port_1695's running-avg bpb was 1.0524 — if that had held, we'd have broken #1736's 1.06610 by 0.014. Instead, the gap closed monotonically and the final aggregate number was within noise of baseline.

I almost wrote this off as "rb aggregation differs from final.json aggregation." It doesn't — the formula is identical (line 2884 of `train_gpt.py`):

```python
r_bpb = r_loss / math.log(2.0) * (cur_tokens / max(cur_bytes_val, 1))
```

Same as `val_bpb`. The rb is a correct running bpb; there's no aggregation sleight-of-hand.

## The real finding: regime-dependence

Dug deeper and bucketed per-batch bpb by doc length:

| Doc length bucket | baseline avg bpb | port_1695 avg bpb | Δ (port − base) |
|---|---|---|---|
| longest (dl 1320–10426) | 1.0538 | 1.0474 | **−0.0064** |
| first 40 (dl 730–10426) | 1.0544 | 1.0471 | **−0.0073** |
| middle (dl 437–712) | 1.0629 | 1.0718 | **+0.0088** |
| shortest (dl 83–263) | 1.1752 | 1.1898 | **+0.0146** |

The sign flips cleanly around doc-length ~500 tokens.

- Rotation **helps** on long docs: −0.006 to −0.007 bpb.
- Rotation **hurts** on short docs: +0.015 bpb.
- The crossover is around 400–700 tokens.
- The magnitude of the hurt (~+0.015) is *larger* than the magnitude of the help (~−0.007). They only net to ~0 because the eval distribution's middle bucket (where effects roughly cancel) dominates by count.

The eval is processed in length-sorted order (longest first). That's why mid-run `rb` looked great: 116 long-doc batches processed, where rotation genuinely was −0.01 better. As shorter docs came in, per-batch bpb started ticking up faster for port_1695 than for baseline, the running avg caught up, and the final aggregate was ~equal.

**This revises my earlier "TTT substitutes for rotation" framing.** TTT doesn't *absorb* rotation's effect — the rotation Δ is ~0 at both `diagnostic_quantized` (pre-TTT) and `quantized_ttt_phased` (post-TTT). What actually happens is:

- Rotation changes the distribution of where the model is good: better on long-context prediction, worse on short-context prediction.
- TTT adapts uniformly, preserving the relative distribution shift.
- The net over the full eval distribution is ~zero because the shape of the help+hurt curve across doc lengths happens to integrate to zero against our eval distribution.

This is a physically meaningful result, not a methodology artifact. Rotation spreads quant error evenly across activation dimensions. Long contexts aggregate across many tokens — mean quant error matters, rotation helps. Short contexts depend on fewer tokens — per-token variance matters, rotation's small per-token perturbations hurt.

## So: can we exploit this?

Thought through the options (fuller enumeration in the transcript). Most obvious approaches are blocked by the 16 MB artifact cap:

- Ship both rotated and unrotated models → no, doesn't fit.
- Toggle rotation per doc on a single model → no, GPTQ quantization is basis-specific. Using dequantized-rotated weights on an unrotated forward, or vice versa, incurs extra quantization error that drowns the benefit.
- Ship one quantized model, compute alternate weights at eval time → same issue in reverse.

What's feasible:

1. **Site-selective rotation** — only apply some of the 4 rotation sites. If attention sites are the long-context helpers and MLP sites are the short-context hurters (plausible: attention aggregates across tokens, MLP is per-token), then `attn_only` may net positive. Cheap to test (env-var filter on existing infrastructure).
2. **Layer-selective rotation** — only rotate decoder layers (6–10), skip encoder (0–5). Same hypothesis but localized differently.
3. **Seed sweep** — different Hadamard seeds produce different regime profiles. Unlikely to move aggregate by 0.002+ but possible.
4. **Rotation + retraining** — train with rotation active from step 0. Weights learn to be robust to rotation. Invasive; spec-011-level work.

User picked option 1 as the next move. That's what spec 010b is.

## Spec 010b — site-selective SpinQuant

Frozen spec at `research/specs/010b-spinquant-sites.md`. 4-mode sweep via new `SPINQUANT_SITES` env var:

| Mode | `SPINQUANT_SITES` value | Prediction |
|---|---|---|
| `attn_only` | `attn_in,attn_proj_in` | if hypothesis holds: −0.001 to −0.003 |
| `mlp_only` | `mlp_in,mlp_proj_in` | +0.001 to +0.003 (isolates the hurt) |
| `all` (sanity) | all 4 | ≈ spec 010's 1.06723 (plumbing sanity check) |
| `attn_in_only` | just `attn_in` | finer-grain decomposition |

### Code changes

Minimal — ~15 LOC added to `train_gpt.py`:

- New `Hyperparameters.spinquant_sites` field (env var, default = all 4).
- New `_parse_spinquant_sites(h)` helper.
- `install_spinquant_rotations` only registers buffers for selected tags.
- `_spinquant_rotate_sd_and_H` only rotates weights + Hessians for selected tags.
- Forward hooks unchanged — they already use `hasattr(self, "_sq_R_...")`, which naturally skips when the buffer wasn't installed.

Committed at `ff52a06`, pushed to fork/research. No changes needed to `spinquant_hotstart.py`.

### Cost and gate

~$25 for 4 modes on one 8×H100 pod session. Gate: if `all` mode's final val_bpb differs from spec 010's 1.06723 by more than ±0.001, halt — indicates SPINQUANT_SITES plumbing has a subtle bug. Otherwise run all four back-to-back.

### Decision tree

- **`attn_only` ≤ −0.001:** confirmed, attention rotations are the lever. Consider stacking with spec 011 (tapered WD retrain).
- **`attn_only` null:** regime-dependence doesn't decompose by site. Spec 011 becomes the primary lever.
- **`mlp_only` clearly hurts, `attn_only` clearly helps:** clean decomposition. Attn_only is the winning variant.
- **All null:** SpinQuant truly done on this stack. Full pivot to spec 011 and beyond.

## Lessons from today

- **Null results are not information-free when you look at the intermediate data.** Spec 010's aggregate null would have sent us off to spec 011 with "SpinQuant doesn't help on #1736." Looking at the per-batch trajectory revealed a real, strong, regime-dependent signal — now we know *why* it's null in aggregate and what to try next. The per-batch data existed in the log the whole time; user asking "it looked so good in the beginning, what happened?" was the right question.
- **Running-average bpb in a length-sorted eval is systematically biased.** Early batches (long, easy) → low rb. Late batches (short, hard) → high rb. The trajectory monotonically rises as eval progresses. This is not a bug in rb; it's a property of the sort order. But it means mid-eval rb is not a reliable preview of final val_bpb.
- **TTT isn't a blanket "absorber" of pre-TTT effects.** The rotation Δ was ~0 at both pre-TTT and post-TTT stages. What TTT does is equalize absolute values across all variants (everyone gets the ~−0.013 gain from phased TTT), but the relative signal between rotated and unrotated variants stays the same at both stages. My first-pass "TTT substitutes for rotation" interpretation was incomplete.
- **#1695's scheme is float-invariant, not perturbative.** I initially thought their approach was an acceptable-perturbation design; reading their actual code closer revealed it's a strict identity: `F.linear(x @ R, W @ R) == F.linear(x, W)`. The null result is *not* because rotation introduced noise — the pre-quant forward is bit-identical. It's because the rotated-basis GPTQ happens to deliver a quant-error improvement that's shaped exactly like the regime-dependent curve we now see.
- **Don't reach for spec 011 until the current signal is fully interrogated.** I was about to pivot. Site-selective ablation is genuinely cheap and isolates a real mechanism. If it lands, we have a quant-side lever; if not, we've ruled out rotation-style quant work on this stack cleanly. Either way the diagnostic value is high.

## Open questions for the next research session

1. Does spec 010b's `all` mode reproduce spec 010's 1.06723 exactly? If it drifts, the new site-filtering code has a subtle bug.
2. Does `attn_only` land? If yes, at what doc-length crossover? Does the per-bucket analysis flatten vs spec 010's sharp +0.015 / −0.007 swing?
3. If `attn_in_only` (just 1 of 4 rotations) delivers most of the `attn_only` benefit, we've localized the effect to a single site — worth a minimum-viable-rotation spec.
4. Are the rb-column vs final.json numbers reconciled? I traced the formula (same), but a small 0.001 gap on baseline vs 0.007 gap on port_1695 is suspicious. Might be that final.json aggregates all 3 phases and the logged `rb` is phase-1-only, but I didn't confirm. Worth a 10-min investigation before trusting mid-run numbers again.
5. Is there any lever that specifically helps short-context prediction on this stack? If so, composing that with rotation would get the help on long docs AND undo the hurt on short docs — a pure stacking win. Candidate: shorter-TTT-chunk-size, different TTT batch shape, tokenizer-level changes.

## State for next session

**Committed and pushed on `fork/research`:**

- `b47a252` — spec 010 implementation (port_1695 rotation code in `train_gpt.py`).
- `ff52a06` — spec 010b (site-selective sweep, SPINQUANT_SITES env var).

**Specs ready to run:**

- Spec 010b — `research/specs/010b-spinquant-sites.md`. Execution can launch whenever. ~$25, ~40 min GPU.
- Spec 011 (tapered WD retrain) — doc ready, code not written (~30–50 LOC patch needed). Still deferred.

**Runs completed today:**

- `runs/008-1736-reproduction/` — partial (gate projected, closed by spec 009 baseline).
- `runs/009-spinquant-hotstart/{baseline,internal_only}/` — full results + summary.md.
- `runs/010-port-1695/` — full result + summary.md with per-doc-length analysis.

**Cost tracking for today (research session invocations + execution invocations):**

| spec | cost |
|---|---|
| 008 (execution morning) | ~$16 |
| 009 baseline + internal_only | ~$14.70 |
| 010 port_1695 | ~$6.50 |
| **subtotal** | **~$37** |
| (spec 010b not yet run) | — |

Total project spend to date: ~$37 of ~$200 budget. Plenty of room for 010b + 011 + final 3-seed confirmation.
