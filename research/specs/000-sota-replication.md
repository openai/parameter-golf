# Spec 000 — SOTA Replication

**Slug:** `sota-replication`
**Created:** 2026-04-19
**Links to idea:** N/A (this is the baseline, not an idea)

## Hypothesis
`train_gpt_sota.py` as-is on 8×H100 NA-1 with seed 42 reproduces the official SOTA submission at **1.0810 ± 0.002 bpb**. If it doesn't replicate, every subsequent Δ is measured against a moving target and further experiments are pointless.

## Baseline
Official leaderboard entry `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` at 1.0810 bpb. Reference: `records/track_10min_16mb/2026-04-09_*/README.md`.

## Expected Δ
Δ = 0 (this is the replication target, not an improvement). Success = landing within ±0.002 of 1.0810.

## Accept criteria
- Single run, 8×H100, seed 42: `val_bpb ∈ [1.079, 1.083]`.
- No NaN, no divergence, wall time within 20% of SOTA submission's ~588s.

## Config diff
None. Run `train_gpt_sota.py` with its shipped defaults (SP8192 tokenizer, 11L / 512d / MLP4×, 3-layer depth recurrence, parallel residuals from layer 7, LeakyReLU², MuonEq-R, GPTQ INT6, SDClip, EMA, TTT enabled, sliding window, brotli-11).

## Code changes
- Branch: `main`
- Commit: **TBD — fill in the current HEAD of `` before running**
- Diff: none

## Hardware ladder
- [ ] 2×H100 mini — **skip**, Exp 24 already validated the SOTA code path on 2×H100
- [ ] 8×H100 official — seed 42

## Seed plan
Single seed (42). Full 3-seed submission is a separate future spec once we have Δ improvements to submit.

## Inputs
- Data: `/workspace/data/datasets/fineweb10B_sp8192/` (verify exact path during interview)
- Tokenizer: `/workspace/data/tokenizers/fineweb_8192_bpe.model`
- Hotstart: none
- Base repo commit: whatever `` currently points to (capture before launch)

## Checkpoints to emit
- Save: final step only (for possible future hotstart experiments)
- State: model + EMA (EMA is what gets evaluated)
- Retention: keep
- Destination: `/workspace/runs/000-sota-replication/checkpoints/step_final.pt`

## Stop-early criteria
- NaN in train loss at any step → kill, mark failed
- val_bpb at midpoint eval > 1.5 → kill, probably broken
- Step time > 2× expected → kill, investigate

## Cost estimate
- 8×H100 NA-1, ~12 min wall: **~$3.50**

## Extra artifacts
None beyond defaults.

## Open questions for interview
- Confirm current `` HEAD is the intended SOTA code (not a stale branch).
- Confirm SP8192 data exists on the NA-1 volume, not just SP1024. If missing, need a data-prep spec first.
- Confirm 8×H100 NA-1 availability before provisioning.
