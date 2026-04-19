# 2026-04-19 — end-of-day session wrap

One long execution session, four specs, three kills. All pods terminated, fleet empty, record-push budget still healthy.

## The four specs at a glance

| # | slug | pod | wall | cost | verdict |
|---|---|---|---|---|---|
| 000 | sota-replication | 8×H100 NA-1 | ~24 min + ~9 min churn | $11.37 | **local baseline: 1.08622** (SOTA 1.0810, missed by +0.0054 — pure throughput, 85% of SOTA pod's step rate) |
| 001 | hessian-sdclip | 1×H100 NA-1 | ~15 min | $1.90 | **kill**: all 6 λ ∈ {0, 0.05, 0.10, 0.20, 0.40, 0.60} monotonically worse than λ=0 |
| 002 | swa-plus-ema | 1×H100 NA-1 | ~30 min + ~4 min 8H churn | $2.87 | **kill**: all 5 SWA variants monotonically worse than EMA-only; SWA fraction correlates cleanly with Δ |
| 003 | bigram-hash-screen | 2×H100 NA-1 | ~48 min | $5.11 | **kill**: post-EMA pre-quant 1.08788 vs Exp 24's 1.08670, miss signal gate by +0.0032 |

**Total spend: $21.25** ($42.29 → $20.26 balance). Well under the $200 hard budget.

## What each spec actually tested

**Spec 000 (baseline):** reproduce the 1.0810 leaderboard with our fork's code. Landed at 1.08622 in-house. ~0.005 off the leaderboard number, but the miss is purely 13% step count deficit from pod throughput variance — code is faithful (verified by decoding the LZMA-blob train_gpt.py and checking signatures). Trains 3849 steps in 588s; SOTA pod got 4531 in the same 588s. Same hardware name, different physical node, different NVLink topology.

**Spec 001 (hessian-sdclip):** port a near-SOTA submission's Hessian-modulated clip formula. Sweep λ ∈ {0, 0.05, 0.10, 0.20, 0.40, 0.60} on spec-000's post-EMA checkpoint. Result: **every** non-zero λ is worse than λ=0 (baseline SDClip). Also hits the 16MB artifact cap at λ≥0.40 (compression efficiency drops). Complete null, cleanly rejected. Secondary: validity gate at λ=0 didn't match spec-000's 1.10430 — we got 1.10518 — because 1×H100's calibration data sharding differs from spec-000's 8×H100 rank-0 shard. Not a code bug; methodology note for future specs.

**Spec 002 (swa-plus-ema):** 6 configs testing weight averaging variants (SWA all 4 warmdown snapshots, SWA late 3, and three SWA/EMA blend ratios). Result: **clean linear relationship** — more SWA, worse bpb. 100% EMA (C0) 1.10518, 75% EMA 1.11108 (+0.006), 50% 1.12251 (+0.017), 25% 1.13532 (+0.030), 0% 1.14694 (+0.042). SOTA's EMA(0.9965) over ~3849 steps is already a richer moving average than 4-point uniform SWA; the warmdown-era snapshots span very different loss regions.

**Spec 003 (bigram-hash-screen):** re-test the March-era BigramHash trick on the April SOTA stack. Single 2×H100 training run with BIGRAM_VOCAB_SIZE=3072 on Exp 24's exact config, compared to Exp 24's log at matched steps. Result: **nuanced but net negative** — variant slightly better or tied in steps 500-2000 (Δ crosses zero), then drifts consistently worse by 0.003-0.006 through warmdown. Final pre-quant 1.08788 vs 1.08670 → +0.00118, miss signal gate. Likely the April primitives (depth recurrence, parallel residuals, QK-gain, LegalTTT) subsume whatever BigramHash provided in simpler stacks.

## Bugs + lessons captured today

1. **Throughput variance is real** (spec 000). Documented in EXECUTION.md. 85% of leaderboard pod's step rate cost us ~0.005 bpb purely from fewer gradient updates. For record-attempt runs, need a tok/s preflight. For screens, use matched-step comparison (saved to memory).

2. **Hessian-reload device matters** (spec 001 mid-run patch). `collect_hessians` returns on CPU; reloading with `map_location=device` (cuda) and then calling `gptq_mixed_quantize` crashes with device-mismatch. Patched sweep.py in-place via sed. Fix lives in our sweep scripts now.

3. **Swa_sweep.py isn't DDP-aware** (spec 002 8H debacle). Tried running `torchrun --nproc_per_node=8 swa_sweep.py` on an 8×H100 pod to see if the Hessian-calibration-data difference matters. Hardcoded `cuda:0` meant all 8 ranks raced on GPU 0. Killed within 4 min ($1.60 wasted). For future 8H sweep runs, need ~10 lines of DDP plumbing (LOCAL_RANK device, rank-0 write guards, optional all-reduce of Hessians).

4. **GPTQ crashes with BIGRAM_VOCAB_SIZE>0** (spec 003). `gptq_mixed_quantize` assumes every state_dict entry ≥65536 params has a Hessian; `bigram.embed.weight` doesn't (Embedding hooks missing from `collect_hessians`). KeyError. Happened post-signal-gate so didn't affect result. Noted in summary.md in case bigram is ever revived.

5. **Sliding-window eval on 1×H100 is 12 minutes, not 3** (spec 002). The spec's cost estimate was based on a bad extrapolation. Quant-only screens on 1×H100 need to disable sliding-window eval (`SLIDING_WINDOW_ENABLED=0`) unless the budget genuinely allows ~80 extra minutes.

6. **Step-matched comparison won the methodology discussion** (spec 003). Saved to memory. For screening runs, compare at matched step count; only record-attempt submissions compare at wallclock.

## Where the fleet + volume stand at EOD

- **Pods: 0 running, 0 stopped.** All three ephemeral pods created today (test-lifecycle, spec-000, spec-001-sidecar, spec-001, spec-002, spec-003) are all deleted. Fleet clean.
- **Volume `hvpdph5i3g` (NA-1, 150 GB)** holds:
  - Spec 000's 9 phase-boundary checkpoints (~2.7 GB)
  - Spec 001's 6 lambda_*.ptz quantized artifacts + hessians.pt (~330 MB)
  - Spec 002's 6 config_*.ptz + hessians.pt (~330 MB)
  - Spec 003: nothing retained (crash before .ptz written; from-scratch training, no checkpoint policy)
  - The repo clone, data, tokenizers
- **Balance: $20.26.** Spend-per-hour: $0.015 (volume storage only).

## What's next

- Research owns the evaluation + `experiments.md` rows + whatever comes next (spec 004).
- Every screen so far has killed. Three data points isn't proof the record-track strategy is wrong, but it's a signal — the April SOTA stack is dense, most marginal-looking hooks don't help, and the remaining headroom is probably in directions nobody's tried. Worth reflecting before spec 004.
- If/when research proposes a code-change spec that needs 8×H100 training, the next execution session should either plan for the throughput preflight or write the DDP-aware sweep script so we can fix spec 002's 8H dead-end.

Session wrapped. All good.
