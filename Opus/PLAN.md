# 3-Day Execution Plan

**Total budget:** $500 RunPod credits
**Deadline:** 2026-04-30 (submission window closes)
**Today:** 2026-04-27

## Bar to clear

- Beat SOTA `val_bpb = 1.0810` by **≥0.005 nats** → target ≤ **1.0760** (3-seed mean)
- Statistical significance **p < 0.01** across seeds {42, 314, 999}
- Train under 600s on 8×H100 SXM
- Eval (sliding + TTT) under 600s
- Artifact ≤ 16,000,000 bytes (decimal MB)

## Why this stack

The PR #1493 SOTA is a long compounded chain. Recent record-to-record deltas have been 0.0006–0.002 nats. To clear 0.005, we need an **orthogonal** improvement, not a hyperparam tweak. The four candidates from highest to lowest EV:

1. **Smarter TTT** — the current TTT trains *all* params with vanilla SGD. Selective param TTT (only adapt fp32 control tensors that survived quantization) is faster, lower variance, and theoretically better-suited to a quantized model.
2. **Code-golf the wrapper** — every byte saved becomes a byte we can spend on weights or precision.
3. **Mixed-bit GPTQ** — per-layer bit allocation by Hessian sensitivity.
4. **MLA + reinvest** — too much surface area to reach parity in 3 days; keep as non-record submission via `train_antigravity.py`.

## Day-by-day

### Day 1 — 2026-04-27 (today) — budget ~$50

**Goal:** Stand up infra. Reproduce SOTA seed=42 to within ±0.0003 of the published 1.08079.

Tasks:
- [ ] Spin up 1×H100 RunPod with the official Parameter Golf template (~$3/hr)
- [ ] Clone repo, download `sp8192` data variant
- [ ] Run `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py` with `SEED=42 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 ...`
  - Note: 1×H100 means `grad_accum_steps=8`; one full run is much slower than 8×H100. Budget ~2 hours wallclock for the full reproduction.
- [ ] Confirm `val_bpb ≈ 1.08079`. If off by more than 0.0003, debug before any other work — this is a leaderboard-validity blocker.
- [ ] Save the trained checkpoint (`final_model.pt`) — Day 2 reuses it for TTT-only experiments without retraining each time.

**Cost estimate:** 1×H100 × ~12h debugging+1 reproduction = ~$36. Buffer to $50.

### Day 2 — 2026-04-28 — budget ~$200

**Goal:** Find a TTT variant that beats the baseline by ≥0.003 nats single-seed (so it has a shot of clearing 0.005 across 3 seeds).

Tasks:
- [ ] Switch to 2×H100 ($6/hr) for faster iteration
- [ ] Reuse Day 1 checkpoint — TTT runs at eval time, no retraining needed for triage
- [ ] Run the experiment matrix in `experiments/` (one .md per experiment):
  - `ttt_selective_scales.md` — TTT only on `q_gain`, `attn_scale`, `mlp_scale`, `skip_weights`, `resid_mix`
  - `ttt_chunk_sweep.md` — chunk sizes {16K, 32K, 64K}
  - `ttt_lr_sweep.md` — {0.001, 0.003, 0.005, 0.008, 0.012} × {2,3,5} epochs
  - `ttt_momentum_reset.md` — reset momentum between chunks
  - `ttt_with_wd.md` — add small weight decay during TTT
  - `ttt_grad_accum.md` — accumulate gradients across chunks before stepping
- [ ] Each experiment: 1 seed, log `val_bpb_ttt`, mark winner/loser
- [ ] By end of day, lock in best config

**Parallel:** I draft a more aggressively golfed code wrapper (target: shave 5KB off the current 16.6KB).

**Cost estimate:** ~30 hours of 2×H100 = $180. Buffer to $200.

### Day 3 — 2026-04-29 — budget ~$230, $20 reserve

**Goal:** Validate winner across 3 seeds on 8×H100. Submit before midnight.

Tasks:
- [ ] 3 × 8×H100 full runs with seeds {42, 314, 999} using the locked Day 2 config
  - Each run: ~12 min total (10 train + ~2 eval) × $20/hr = ~$4 each = $12 total compute
  - Add ~30 min of pod warm-up overhead per run = $10 each ≈ $30 total
  - Realistic: $40 for 3 seeds
- [ ] If 3-seed mean clears 1.0760 with std ≤ 0.0007 (so p<0.01 by t-test vs 1.0810): proceed to PR
- [ ] If marginal (mean 1.0765–1.0775 or std too high): one more 3-seed run with seeds {7, 11, 13} to either confirm or kill
- [ ] Write the submission README + `submission.json` modeled on PR #1493's
- [ ] Open the PR on `openai/parameter-golf` with all 3 train logs attached

**Reserve $20:** if everything goes sideways, one final 8×H100 run to test the most-promising untried idea.

## Stopping rules

- If Day 1 reproduction is off by >0.0005 BPB: **stop and debug** before spending more.
- If Day 2 noon and no TTT variant has hit ≥0.002 single-seed gain: **pivot remaining $$ to mixed-bit GPTQ**.
- If Day 3 morning and 3-seed mean is >1.0790: **abandon record attempt**, focus the remaining hours on shipping `train_antigravity.py` as a polished non-record submission.

## Parallel non-record track (low priority, low cost)

`train_antigravity.py` (MLA + 3.5× MLP + Int6 QAT) — keep this alive as a non-record submission to the unlimited compute track. Even at modest BPB it goes on the record as a creative submission. Touches none of the leaderboard budget if we don't spend extra compute on it; we package whatever the latest `EXPERIMENTAL_LOG.md` shows.
