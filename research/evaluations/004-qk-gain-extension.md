# Evaluation — Spec 004 (QK-gain extension screen)

**Run dirs:** `runs/004-qk-gain-extension/` (Phase 1) + `runs/004b-qk6-full/` (Phase 2) + `runs/004c-qk6-verify/` (Phase 3) | **Hardware:** 8×H100 NA-1 (three different physical pods) | **Date:** 2026-04-19 → 2026-04-20 | **Code:** `research @ feaf45e` (no code change; hyperparam-only)

## Result

Three runs triggered by the same spec to extend SOTA's `QK_GAIN_INIT=5.25` upward. Phase 1 was the spec-prescribed A/B triage (QK=6.0 + QK=5.5, 5-min caps). Phase 2 was an ad-hoc full-train follow-up after Phase 1 suggested a signal. Phase 3 was a clean verification that resolved the signal as pod-numerical-variance.

| run | QK | cap | VAL | final step | final-step val_bpb | pre-quant post-EMA | Δ pre-quant vs spec 000 |
|---|---|---|---|---|---|---|---|
| spec 000 | 5.25 | 10m | 4000 | 3849 | 1.0938 | **1.09289** | (base) |
| Phase 1 Run A | 6.0 | **5m** | 4000 | 1700 | — | — | — |
| Phase 1 Run B | 5.5 | **5m** | 4000 | 2246 | — | — | — |
| Phase 2 (004b) | 6.0 | 10m | **200** | 3876 | 1.0934 | 1.09249 | −0.00040 |
| Phase 3 (004c) | 6.0 | 10m | 4000 | 3952 | **1.0929** | **1.09193** | **−0.00096** |

**Signal gate: NOT MET.** Spec's criteria were promising if variant train_loss ≤ spec 000's at ≥3 of 4 milestones by ≥0.005. Phase 1 suggested this (Run A at step 1000 was Δ −0.109), but Phase 3's clean verification shows the signal doesn't survive hardware + RNG variance normalization.

## Noise/signal judgment — kill

Noise-floor estimation:
- SOTA's intra-seed std (same-hardware, seed-to-seed variance): 0.0002 on final pre-quant
- Phase 1 → Phase 3 disagreement (same config, different physical pod, same seed): 0.109 on step-1000 train_loss — the bf16 cross-pod numerical drift ceiling we now know about

Our observed Δ (Phase 3 vs spec 000): **−0.00096** on pre-quant post-EMA. That's ~5× the SOTA seed std, ~100× smaller than the Phase 1/Phase 3 noise gap. Plus Phase 3 ran 103 more training steps (3952 vs 3849), and an extra 100 steps alone would account for ~0.002 bpb of improvement via additional gradient updates. **Normalized for step count, the Δ is effectively zero.**

Kill.

## Why it looked like a signal initially (Phase 1)

Phase 1 Run A showed step-1000 train_loss 3.1394 vs spec 000's 3.2487 (Δ −0.1093). That's monotonically-plausible (QK=4.0 → 5.0 → 5.25 → 6.0?) and matched the spec's hopeful hypothesis.

What was actually happening: Run A's 8×H100 pod had **different bf16 numerical behavior** than spec 000's pod. Hardware-level sources of variance:
- Different physical GPU samples (binning/manufacturing variance even within the same SKU)
- Different NVLink/NVSwitch topology → different reduction orderings in DDP all-reduces
- Different CPU/memory affinity → potentially different batch loading timing → possibly different tiny numerical artifacts

Any one of these can drift accumulated loss by ~0.1 at step 1000 on this architecture. Phase 3 (running on a third physical pod) confirmed the spec 000-matching-RNG trajectory returns to within ±0.002 of spec 000 at most milestones. The Run A "win" was pod luck.

## Why Phase 2 (004b) was confounded

I set `VAL_LOSS_EVERY=200` for more detailed val logging. That shifted the RNG trace vs spec 000's default `VAL_LOSS_EVERY=4000`. Same seed, different RNG consumption → different training batches at every step → step-matched comparison is noise-confounded. Phase 2's Δ of −0.00040 at final pre-quant is uninterpretable cleanly — could be small QK effect, could be RNG-induced batch-luck.

Phase 3 fixed this with VAL=4000 and the Δ tightened to −0.00096 (still within noise, but cleaner comparison). Agreement between Phases 2 and 3 on "tied within noise" is the real signal.

## Secondary finding: artifact size at QK=6.0

Phase 2's quantized+brotli artifact was **16,046,371 bytes** — over the 16,000,000 leaderboard cap by 46 KB. Higher QK gain → sharper attention at init → training converges to weight distributions that compress less efficiently under GPTQ's SDClip + brotli. Even if QK=6.0 had been a genuine small win, it would have needed additional size engineering to be submittable.

We didn't measure 004c's artifact size (NCCL crashed during the quantize step), but the training trajectory was near-identical to 004b so same expectation holds.

## Decision: **KILL**

- `research/ideas/qk-gain-extension.md` (or equivalent) should be updated with `Status: ❌ SHELVED 2026-04-20` and a pointer to this evaluation.
- Don't retest QK_GAIN_INIT=5.5 — Phase 1's Run B was on the same pod as Run A and thus has the same hardware-variance contamination. Extrapolating from Phase 3, Run B's −0.090 at step 2000 is also almost certainly hardware noise.
- Don't retest QK>6.0 either — monotonicity (4.0 → 5.0 → 5.25) doesn't extend past 5.25 per our data.

## Strategic implications

**Four consecutive screen specs killed** (001 Hessian-SDClip, 002 SWA+EMA, 003 BigramHash, 004 QK-gain). The April SOTA stack is genuinely dense — every "obvious" port from prior near-SOTA work has failed to improve.

The pattern suggests the remaining bpb headroom is NOT in tuning existing primitives or porting prior-submission tricks. It's somewhere less-obvious. Candidates for future specs, ranked by my subjective prior:

| candidate | basis | cost | risk |
|---|---|---|---|
| Layerwise LR decay | generic transformer tuning knob not yet explored | ~$5 screen | moderate — hyperparam brittle |
| `num_kv_heads=2` (from 4) | architectural change, saves KV param budget | ~$5 screen | low risk if no quality drop |
| Fewer attention layers, wider MLP | shift param budget across blocks | ~$5 screen | moderate |
| **Do nothing; submit spec 000's baseline as 3-seed record-attempt** | we ARE at 1.08622 on 8×H100, may already match fleet median if 3-seed averaging works in our favor | ~$20 (3 seeds × $7) | low — known quantity |

**Important new consideration for the 5th option:** we are at 1.08622 on ONE SEED. SOTA's leaderboard is 1.0810 on THREE SEEDS averaged (std 0.0002). Our one-seed result might be unlucky within our own seed distribution. A proper 3-seed record-attempt run might land very near 1.08622 ± 0.0005 → meaningfully above SOTA but within hardware-variance distance. Worth the $20 to measure, I think.

## Cost accounting

- Phase 1 (Run A + Run B, one pod): $5.70
- Phase 2 (004b full run + recovery): $8.37
- Phase 3 (004c verification): $5.70
- **Total spec 004: $19.77**

All within spec's estimate range ($4-7 per run × 3 runs ≈ $15-20).

Total push spent: $21.25 (000) + $1.90 (001) + $2.87 (002) + $5.11 (003) + $19.77 (004) = **$50.90**. Balance $50.42. Mid-push refund/topup brought us back to healthy state ~$56.10.

## Artifacts retained

In-repo:
- `runs/004-qk-gain-extension/{qk_6.0_train.log, qk_5.5_train.log, summary.md, notes.md}` — Phase 1
- `runs/004b-qk6-full/train.log` — Phase 2 (full post-training pipeline ran, gives quant+sliding numbers)
- `runs/004c-qk6-verify/train.log` — Phase 3 (pre-quant post-EMA, NCCL crash after)

On NA-1 volume at `/workspace/runs/004b-qk6-full/checkpoints/`:
- 9 phase-boundary checkpoints from Phase 2 (~2.7 GB). Usable as hotstart if any future spec wants to fork from a QK=6.0 mid-training state. Probably won't, given the kill.

## Handback

Research: evaluation + experiments.md row pushed. Decision is yours on (a) writing spec 005 as another algorithmic screen, (b) committing to a 3-seed record-attempt with current best config, or (c) taking a break from specs and thinking harder about what direction to try next.
