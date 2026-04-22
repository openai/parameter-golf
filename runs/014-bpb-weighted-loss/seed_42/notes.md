# Spec 014 seed_42 — execution notes (screening run)

**Run dir:** `runs/014-bpb-weighted-loss/seed_42/`
**Commit:** `ab6a131` on `exp/bpb-weighted`
**Date:** 2026-04-21
**Pods:**
- Smoke: `qkbgrhvx12lkuk` (2×H100 SXM AP-JP-1 SECURE) — DELETED after smoke pass
- Full: `9l34ork0rt5vc0` (8×H100 SXM AP-JP-1 SECURE) — STOPPED after endpoint
**Mode:** screening — killed at `stopping_early: wallclock_cap` via watcher.

## Status

**Training completed cleanly.** Hit wallclock cap at step **4797** / 596,043 ms. No NaN, no destabilization.

`bpb_weighted_loss:` config line confirmed: `enabled=True mean_weight=4.959 min=1 max=15`. Mean weight 4.959 sits squarely in the spec's predicted "~3-5 for SP8192" band; max 15 is the longest surface piece in the SP8192 vocab.

## Endpoint metrics — clear regression

| metric | spec 008 (#1736 repro) | spec 014 (BPB-weighted) | Δ (014 − 008) |
|---|---|---|---|
| stopping_early step | 4828 | **4797** | −31 |
| stopping_early train_time | 596.18 s | 596.04 s | ≈ equal |
| **mid val_bpb @ 4000** | **1.1110** | **1.1743** | **+0.0633** |
| **endpoint val_bpb** | **1.0697** | **1.1316** | **+0.0619** |

**Verdict (per spec's accept criteria):** Δ > +0.001 = regression. We're 62× past that threshold. Lever does not transfer to this stack.

## Train_loss curves (informational only — different metrics)

Spec 008 logs **uniform** CE; spec 014 logs **byte-weighted** CE. Numbers not directly comparable. Showing for completeness — Δ values are NOT meaningful.

| step | spec 008 (uniform) | spec 014 (weighted) | "Δ" (artefact of metric mismatch) |
|---|---|---|---|
| 1 | 9.0180 | 9.0245 | +0.0065 (≈0 — untrained model, weighting irrelevant) |
| 500 | 2.5807 | 3.1894 | +0.6087 |
| 1000 | 2.8105 | 3.3902 | +0.5797 |
| 1500 | 2.6434 | 3.2335 | +0.5901 |
| 2000 | 2.6723 | 3.2188 | +0.5465 |
| 2500 | 2.5580 | 3.1397 | +0.5817 |
| 3000 | 2.5662 | 3.1184 | +0.5522 |
| 3500 | 2.5716 | 3.1116 | +0.5400 |
| 4000 | 2.4095 | 2.9205 | +0.5110 |
| 4500 | 2.2803 | 2.7418 | +0.4615 |

The "Δ" column is dominated by the metric difference (5x amplification on multi-byte tokens), not by training quality. As model learns multi-byte tokens, weighted loss converges toward uniform; "Δ" tightens from +0.61 → +0.46. None of this is informative about the lever.

## Why the val_bpb regression is real

Spec author warned: "destabilizes training on large vocabs (GPT-2 50K). SP8192 is our risk zone." We did not see destabilization (no NaN, smooth curve). What we saw is more subtle: **the byte-weighted gradient direction is structurally different from uniform CE, and Muon's LR schedule is tuned on uniform CE.** With weighted gradients concentrated on multi-byte tokens, the effective LR / step-size profile is mismatched, and the model under-trains at endpoint.

The +0.0619 endpoint Δ is consistent at the +0.0633 mid-train Δ — a constant gap, not a gap that closes. Suggests the entire training trajectory is shifted, not a transient effect that catches up.

## Smoke addendum

2×H100 smoke ran cleanly through 500 steps in ~3.5 min. Boot line confirmed config. Train_loss step-1 = 9.0244 (vs spec 008's 9.0180, +0.006) — close, exactly as predicted by spec ("close to 9.0180"). Loss decreased monotonically through all 500 steps, no NaN, no oscillation. Per spec destabilization-kill criteria, all pass. Smoke cost ~$1.

**However:** smoke train_loss @ step 500 was 3.27 (high vs spec 008's 2.58) — this looked alarming at the time but turned out to be the metric-mismatch artifact, not a destabilization signal. Lesson: **do not interpret 014's train_loss against spec 008** — it's a different function. Only val_bpb is comparable.

## Artifacts

- `train.log` (6 KB) — full training log up through stopping_early.
- `screen_endpoint.txt` — captured snapshot of endpoint val + stopping_early line.
- `launch.out` (empty — torchrun went straight to train.log).

## Cost accounting

| item | cost |
|---|---|
| Polling for capacity (~30 min wall, no spend) | $0 |
| 2×H100 smoke (~5 min on $5.98/hr) | ~$0.50 |
| 8×H100 idle while smoke ran (~5 min on $23.92/hr) | ~$2.00 |
| 8×H100 full screening run (~12 min wall) | ~$5.00 |
| **Total spec 014 spend** | **~$7.50** |

The "8H idle while smoke ran" was the cost of parallelization — accepted upfront for wallclock savings. Could've sequentialized for $0 idle but +5 min wallclock.

## Things that went right

- Smoke caught nothing scary (no destabilization), so 8H launch was justified.
- Watcher (correct pattern this time) killed torchrun cleanly within 5s of stopping_early.
- 30s monitor with matched-step Δ + apples-to-apples val_bpb header was useful for live judgment — surfaced the +0.0633 mid-train Δ immediately, before we'd burned the rest of training.
- Spec's pre-registered expectations ("step-1 close to 9.0180") were a useful sanity check — confirmed at 9.0245.

## PR #1519 formula verification (post-run sanity check)

Verified spec 014's loss formula matches PR #1519 exactly: `(per_token_ce * byte_weight).sum() / byte_weight.sum()` (with a defensive `clamp_min(1.0)` on the denominator that never triggers). Both codebases also keep eval uniform: PR gates on `self.training`; spec 014's `eval_val()` calls `forward_logits` directly and computes its own uniform CE. No formula or metric-mismatch bugs.

The regression is consistent with the PR author's **own pre-registered failure mode**: "Verified it does NOT work with large vocabularies (GPT-2 50K) where extreme byte lengths destabilize training." Our SP8192 (max byte weight 15) sits between PR's tested SP1024 (max ~8, lever lands) and GPT-2 50K (lever breaks). PR landed −0.0194 on a 1.1340 baseline; we got +0.0619 on a 1.0697 baseline (8x stronger). Both vocab-size and baseline-strength bias against transfer; both materialized.

## Things to flag for research

1. **The "train_loss not directly comparable" issue caught us briefly** — at smoke completion the +0.69 gap looked like destabilization for ~30 seconds before the metric-mismatch realization. If future specs change the loss definition, the spec should explicitly state "ignore train_loss comparison; trust val_bpb only" up front.
2. **The stack-bias hypothesis bites again.** Spec author's pre-registered honest-self-check warned: "Three null/regression results in a row (011 GradPower, 013 BigramHash, hypothetical 014 null) would suggest #1736 is already well-tuned, incremental ports from other stacks don't stack on top." That hypothesis just got a lot of weight.

## Handback

Spec 014 (BPB-weighted CE loss) is a **clear regression on #1736** — endpoint val_bpb +0.0619 vs spec 008 baseline. Lever does not transfer. Per spec accept criteria, **shelve**.

Three open questions for research:
1. Is there a "BPB-weighted but with retuned Muon LR" version worth testing? Spec mentions effective-LR shift as a possible cause; smaller LR on multi-byte gradients could fix it. Cost: ~$25 for a 3-point LR sweep.
2. Should the "honest self-check" branch (b/c) be acted on? — commit 3-seed official on plain spec 008, or pivot to AWQ from GPTQ.
3. Worth running the BPB-weighting on TTT's loss too, even though training-time weighting failed? Probably no, but cheap to confirm in the eval-only path.

Pod `9l34ork0rt5vc0` stopped, container preserved for ~end-of-day. Will probably terminate at session end.
