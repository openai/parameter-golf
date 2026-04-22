# Spec 013 seed_42 — execution notes (screening run)

**Run dir:** `runs/013-bigram-hash/seed_42/` (local + JP volume `jlxvxeiol4:/workspace/runs/013-bigram-hash/seed_42/`)
**Commit:** `66e57bf` on `exp/bigram-hash`
**Date:** 2026-04-20
**Pod:** `079ww0a7hwxf29` (8×H100 SXM AP-JP-1, $23.92/hr) — STOPPED
**Mode:** screening — killed at `stopping_early: wallclock_cap` via watcher; no EMA/GPTQ/sliding/TTT.
**Smoke:** SKIPPED per user direction (JP 2×H100 capacity unavailable; user accepted the bug-discovery risk to skip and go 8×H100 direct).

## Status

**Training completed cleanly.** Hit wallclock cap at step **4833** / 596,052 ms. No NaN, no crash.

`bigram_hash:` config line confirmed at boot:
`bigram_hash: enabled=True buckets=16384 dim=32 primes=(36313,27191)`

Skip-smoke risk paid off: 110 LOC of new code ran first try at 8×H100 with no shape/optimizer/GPTQ-hook bugs. (Spec 011's commit `8d54854` was not so lucky.)

## Endpoint metrics

| metric | spec 008 (#1736 repro) | spec 013 (bigram) | Δ (013 − 008) |
|---|---|---|---|
| stopping_early step | 4828 | **4833** | +5 |
| stopping_early train_time | 596.18 s | 596.05 s | ≈ equal |
| step-endpoint val_loss | 2.347 | 2.3465 | ≈ equal |
| **step-endpoint val_bpb (bare)** | **1.0697** | **1.0722** | **+0.0025** |
| step-4000 mid-train val_bpb | 1.1110 | 1.1144 | +0.0034 |

## Matched-step train_loss curve

| step | spec 008 | spec 013 | Δ |
|---|---|---|---|
| 1 | 9.0180 | 9.0118 | −0.0062 |
| 5 | 7.8437 | 7.8139 | −0.0298 |
| 500 | 2.5807 | 2.6130 | **+0.0323** (early peak) |
| 1000 | 2.8105 | 2.8255 | +0.0150 |
| 1500 | 2.6434 | 2.6610 | +0.0176 |
| 2000 | 2.6723 | 2.6878 | +0.0155 |
| 2500 | 2.5580 | 2.5692 | +0.0112 |
| 3000 | 2.5662 | 2.5826 | +0.0164 |
| 3500 | 2.5716 | 2.5737 | **+0.0021** (curve crossing point) |
| 4000 | 2.4095 | 2.4136 | +0.0041 |
| 4500 | 2.2803 | 2.2872 | +0.0069 |

**Trajectory shape:** spec 013 starts +0.0323 above spec 008 at step 500 (RNG-stream divergence + cold zero-init bigram), gap closes monotonically to +0.0021 at step 3500 as the bigram embedding learns, then re-widens slightly to +0.0069 at step 4500. Endpoint Δ +0.0025 val_bpb.

## Observation (no interpretation — research's call)

The spec's pre-registered expectation was −0.001 to −0.003 endpoint val_bpb Δ. Observed +0.0025 — outside the predicted band on the wrong side, but within the per-seed noise floor (~±0.001-0.002 for #1736-class runs). Whether this counts as a null or a small regression is for research to decide.

The mid-training curve crossing the gap at step 3500 (+0.0021) is consistent with the spec's "bigram embedding learns to contribute" narrative — the lever IS doing something. It just doesn't outpace whatever advantage spec 008's RNG path had at step 500.

## Artifacts

- `train.log` (6 KB) — full training log up through stopping_early.
- `screen_endpoint.txt` — captured snapshot of the last training rows + endpoint val + stopping_early line.
- `launch.out` (empty — torchrun went straight to train.log).

No `final.json`, no GPTQ artifact, no checkpoints — screening mode.

## Cost accounting

| item | cost |
|---|---|
| Polling loop (no cost — waits for capacity) | $0 |
| Pod start (already provisioned, container preserved from spec 011) | $0 |
| 8×H100 full screening run (~12 min wall: ~3 min compile + ~10 min training + minimal post-train before kill) | ~$5 |
| **Total spec 013 spend** | **~$5** |

## Things that went right

- Fix-fast retry of 011 plumbing held: `DATA_DIR=/workspace/data`, pyminify preinstalled, watcher pattern `"stopping_early: wallclock_cap"` correct, kill fired immediately.
- 110 LOC bigram code ran clean at 8×H100 first try (skip-smoke gamble paid off this time).
- Live matched-step monitor showed the full train_loss-Δ trajectory in real time — useful for the research-side narrative.

## Handback

Training healthy, endpoint val_bpb +0.0025 vs spec 008 (likely null, small bias on the wrong side). Bigram does appear to "learn something" mid-training (gap closes from +0.0323 → +0.0021 between step 500 and 3500) but doesn't beat baseline by endpoint.

Research to decide:
- Null vs small regression? Compare to per-seed std (~±0.001-0.002 expected).
- Worth a 3-seed re-test for variance estimate, or shelve?
- Worth investigating why early-step Δ is +0.0323 (RNG path? Init scale?) before any retry?
- Per spec's open question 2: the hash's CaseOps collision profile — maybe verify before any retry?

Pod `079ww0a7hwxf29` stopped, container preserved if research wants follow-up.
