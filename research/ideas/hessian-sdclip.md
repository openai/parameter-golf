# Hessian-Aware SDClip (λ-weighted)

**Status:** candidate (strongest of the near-SOTA port set)
**Expected Δ:** +0.0002 to +0.0010 bpb (author-reported on weaker baseline; transfer to SOTA uncertain)
**Source:** `records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/README.md`, section "Hessian-Aware SDClip".

## Idea
SOTA's SDClip picks the per-row quant clip threshold from `c_i = k · σ_i` (std of row `i`). Near-SOTA modulates this by a per-row importance `r_i` derived from the GPTQ Hessian diagonal:

```
c_i = k · σ_i · [1 + λ (r_i − 1)]
```

with `λ = 0.175`. `r_i` is the normalized Hessian-diagonal row importance (mean 1 across rows of a matrix).

Effect: high-importance rows (large Hessian diagonal) get a slightly looser clip = finer quantization; low-importance rows get a tighter clip = more compressible after Brotli. The near-SOTA author reports a sweet spot at λ=0.175; higher λ raises entropy enough that Brotli compresses worse, erasing the gain.

## Why it might help
- GPTQ already computes the per-matrix input Hessian (`train_gpt_sota.py:763-806`, `collect_hessians`). The signal is free.
- SOTA's uniform `k · σ_i` treats all rows as equally important. Real importance varies — the near-SOTA author found group-level Hessian traces very stable across seeds (r=0.997), suggesting the signal is reliable.
- Trades entropy for rounding error at the row level. Brotli's statistical model benefits from concentrating entropy where the model tolerates it.
- Architecture-orthogonal: nothing about the trained weights changes.

## Hotstart screening plan (the killer feature)
**This is a post-training quant change.** No training steps are redone.

- **Hotstart from:** `runs/000-sota-replication/checkpoints/ckpt_final_pre_ema_step~4550.pt`.
- **Pipeline re-run:** EMA application → GPTQ calibration → SDClip (with λ) → INT6 quant → Brotli → eval. Nothing else.
- **Control:** same pipeline, λ=0 (== current SOTA behavior).
- **Wall time per run:** 2-3 minutes on 1×H100.
- **Cost per run:** <$0.50.
- **Sweep:** λ ∈ {0.10, 0.15, 0.175, 0.20, 0.25, 0.30} ≈ 6 runs ≈ $3 total.
- **Noise floor:** near-zero. Same weights, deterministic calibration (given fixed calibration batch), only the clip formula differs.
- **Promotion threshold:** any λ producing Δ ≥ 0.0003 over control is worth keeping. Δ ≥ 0.0005 is likely a real record-contributor. Because screen noise is essentially zero, even small Δ is signal.

## Special case: screen ≈ ship
Unlike the other candidates, a **successful screen IS a valid submission** for this idea. The submission pipeline and the screen pipeline are the same — just one hyperparam differs. The only thing a full-training run would add is: training weights that were optimized *knowing* λ would be applied at quant time. That's a second-order effect and probably small.

## Code-change sketch
Locate SDClip code in `train_gpt_sota.py`. Modify the clip threshold computation to:

```python
hessian_diag = hessians[name].diagonal()  # already computed by collect_hessians
r = hessian_diag / hessian_diag.mean()     # normalized importance, mean=1
c = k * row_std * (1 + hessian_lambda * (r - 1))
```

Add env var `HESSIAN_CLIP_LAMBDA` (default 0.0 = baseline). Reference impl: `records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt.py`.

## Risks / open questions
- **Transfer risk:** the near-SOTA result was on a 1.0835 base stack. On a 1.0810 base, the Δ may be smaller or absent if SOTA's other quant improvements already soak up the margin.
- **Calibration-data sensitivity:** does λ change what GPTQ picks for the per-column error correction? If yes, there may be second-order interactions we haven't thought through.
- **Stacking with per-group bit allocation:** both modify the quant pipeline. If both screen well, test them stacked in a single screen before any full run.
- **λ sweet-spot drift:** the near-SOTA sweet spot was 0.175 on their architecture. SOTA has different depth recurrence + parallel residuals — the sweet spot may move. Sweep is cheap, do it.

## If this works
- Best single-candidate EV given its cost: +0.0005 at essentially zero incremental cost.
- Stacks cleanly with Candidates 2 (two-phase recurrence) and arguably with Candidate 3 (per-group bit allocation, if carefully tested together).
- Probably can ship as a record attempt directly from spec-000 checkpoints, no full retrain needed.
