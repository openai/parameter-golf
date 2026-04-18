# Evaluation — Spec 002 (SWA + EMA blend screen)

**Run:** `runs/002-swa-plus-ema/` | **Hardware:** 1×H100 NA-1 | **Date:** 2026-04-19
**Code:** `exp/swa-plus-ema` @ `46c2a92` | **Hotstart:** spec-000 post-recurrence checkpoints (steps 1500, 2275, 3412, 3849)

## Result

6 configs on 1×H100. Shared Hessian from C0's weights. Quant-only for C1-C5 (sliding-window skipped — see below).

| cfg | description | val_bpb_quantized | Δ vs C0 |
|---|---|---|---|
| C0 | EMA-only (control) | **1.10518** | (base) |
| C4 | 0.25 SWA + 0.75 EMA | 1.11108 | +0.006 |
| C3 | 0.50 SWA + 0.50 EMA | 1.12251 | +0.017 |
| C2 | SWA late 3 (2275, 3412, 3849) | 1.12273 | +0.018 |
| C5 | 0.75 SWA + 0.25 EMA | 1.13532 | +0.030 |
| C1 | SWA all 4 (1500, 2275, 3412, 3849) | 1.14694 | +0.042 |

Plus C0's full-pipeline: `val_bpb_sliding = 1.08869`. Other configs didn't get sliding.

**Signal gate (any non-C0 config with Δ_quant ≤ −0.0003): NOT MET.** All 5 non-C0 configs are positive Δ.

## Noise/signal judgment — clean kill

Not noise. **Monotonic linear worsening with SWA fraction:**

| EMA share | SWA share | Δ vs C0 |
|---|---|---|
| 100% | 0% | 0 |
| 75% | 25% | +0.006 |
| 50% | 50% | +0.017 |
| 25% | 75% | +0.030 |
| 0% | 100% | +0.042 |

Effect is ~4.2% bpb at 100% SWA — orders of magnitude above SOTA's 0.0002 std. Reproducible, directional, clean.

C2 (SWA of late 3 only) at +0.018 is nearly identical to C3 (half-EMA-blend SWA of all 4 at +0.017). Which tells us: dropping the earliest snapshot (step 1500) is roughly equivalent to adding 50% EMA weight. Either way, more EMA = better.

## Why it didn't work (hypotheses — for learning, not action)

1. **EMA decay 0.9965 is already the right averaging.** Effective window ~285 steps over 3849-step run = fine-grained weighted average of the last 7%. A 4-point uniform SWA can't beat that; it just adds noise from earlier, less-trained checkpoints.

2. **Our snapshots span incompatible loss-landscape regions.** Step 1500 is where recurrence just activated and LR is still high. Step 3849 is near-zero LR. Classical SWA assumes snapshots from a flat low-LR plateau; we don't have that. Dropping 1500 (C2) helped, but not enough.

3. **Quant-penalty amplification.** Averaged weights may have lower magnitude variance → tighter SDClip threshold → larger quant error per row. Didn't measure directly but the quant bpb pattern is consistent with this.

None of these hypotheses are actionable — we aren't going to re-run spec 000 with denser post-convergence checkpoints just to retry SWA. Kill the candidate.

## Secondary finding — sliding eval is ~12 min on 1×H100, not 3 min

Spec estimated sliding-window eval at ~3 min per config. Actual: **717 seconds (~12 min)** on 1×H100. 4× over estimate.

Projected full sweep with sliding: ~90 min / ~$4.50 instead of 18 min / ~$1. Execution pivoted mid-run to quant-only for C1-C5, using `SLIDING_WINDOW_ENABLED=0`. Correct call — sliding would have confirmed the same direction on clearly-killed configs.

**Lesson for future 1×H100 specs:** budget sliding at ~12 min/config, not 3. If more than 2-3 configs, consider skipping sliding in the screen and re-evaluating only the winner(s). Updated mental model for any future post-training screen spec.

## Secondary finding — our sweep scripts aren't DDP-aware

Execution attempted an 8×H100 A/B of C0 to see if pod shape affects the bpb. Result: all 8 ranks raced on GPU 0 because `swa_sweep.py` hardcodes `device = torch.device("cuda", 0)`. GPU 0 hit 80 GB, other 7 idle. Aborted within 4 min ($1.60 wasted).

**For future sweep scripts that need multi-GPU:** ~10 lines of DDP machinery (LOCAL_RANK device selection, rank-0 write guards, Hessian reduce-or-keep-local decision, per-rank compile cache). Not urgent — single-GPU is fine for most screens — but noting as a pattern-level limitation.

## C0 validity — bitwise-exact vs spec 001's λ=0

C0 reproduced spec 001's λ=0 quantized bpb **bitwise-exactly**: 1.1051789806396541 both runs. Confirms our sweep pipeline is deterministic given fixed checkpoint + seed + calibration slice. Absolute number is ~+0.0009 off spec 000's 8×H100 1.10430 baseline — expected 1-GPU vs 8-GPU Hessian-calibration offset, already documented in spec 001's evaluation.

## Cost accounting

- Spec estimate: ~$1.70 base, ~$1.20 early-kill.
- Actual sweep cost: **~$1** (reused spec 001's already-stopped pod; no cold provisioning).
- 8×H100 A/B aborted: **$1.60**.
- **Total: ~$2.60.** Under the spec ceiling.

## Decision: **KILL**

- Idea `research/ideas/swa-plus-ema.md` updated with `Status: ❌ SHELVED 2026-04-19`.
- Don't revisit with different snapshot subsets, different blend ratios, or different SWA periods. The linear-in-SWA-fraction pattern means the mechanism is incompatible with our stack — more experiments just confirm the kill.
- A fundamentally different formulation (e.g. "train to post-convergence plateau, then take dense SWA snapshots over a narrow window") would require re-architecting training and is out of scope for the record push.

## Strategic implications

**Second post-training candidate dead.** Spec 001 (Hessian-SDClip) killed yesterday; spec 002 (SWA+EMA) killed today. Two big cheap post-training candidates off the board, both with clean monotonic-worsening signals.

**Post-training ceiling is demonstrably very low.** SOTA's GPTQ + SDClip + EMA + Brotli pipeline is well-tuned enough that both "tweak SDClip with Hessian info" and "blend an SWA state into EMA" make things strictly worse. This reorients our remaining $180 budget:

- **Training-time candidates are now the only credible path to record.**
- Spec 003 (BigramHash screen) becomes the load-bearing experiment.
- If BigramHash also kills, fallback is layerwise-LR-decay; then architecture knobs.

## Next steps

1. **Execute spec 003 (BigramHash screen).** Already frozen, ready for pod. ~$4. Answers the biggest remaining question.
2. **If BigramHash promotes:** spec 004 = full 8×H100 with `BIGRAM_VOCAB_SIZE=3072` + budget-fit engineering (MLP ratio cut or num_kv_heads=2 to fit 16MB).
3. **If BigramHash kills:** write new spec for layerwise LR decay, possibly combined with a small architecture change (num_kv_heads=2). Cost ~$10 per full run.
4. **Budget check:** $14.90 (000) + $1.90 (001) + $2.60 (002) = $19.40 spent. ~$180 remaining, ~10-11 days.

## Artifacts retained

On NA-1 volume at `/workspace/runs/002-swa-plus-ema-1h-c0/`:
- `hessians.pt` — 232 MB. Reusable for any future Hessian-based experiment on spec-000's post-EMA weights.
- `quantized_C0.ptz` through `quantized_C5.ptz` — ~16 MB each. Retained if any follow-up wants sliding or TTT on a specific config; probably moot given the kill decision.

In repo at `runs/002-swa-plus-ema/`:
- `config_C0.json` through `config_C5.json`.
- `summary.md` (execution), `notes.md` (execution narrative), `sweep.out`.
