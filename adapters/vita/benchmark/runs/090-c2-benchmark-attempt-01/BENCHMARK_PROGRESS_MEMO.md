# Benchmark Progress Memo — 090/C2 Track

Status: non-claiming benchmark iteration log
Date: 2026-03-24

## Attempt01 (baseline)

- Source: 090/C2 lead selected from VITA workflow
- Delta policy: no leaderboard-inspired delta yet
- Result:
  - `final_int8_zlib_roundtrip_exact val_bpb: 1.23814533`
  - `final_int8_zlib_roundtrip_exact val_loss: 2.09055653`
- Evidence:
  - `evidence/metric_8gpu_attempt01.txt`
  - `evidence/train_8gpu_attempt01.log`
  - `evidence/artifact_summary_8gpu_attempt01.txt`

Interpretation:
- first real benchmark baseline established
- full 8-GPU benchmark path executed successfully

## Attempt02 (single delta)

- Base: Attempt01
- Single change: `WARMDOWN_ITERS=3500`
- Result:
  - `final_val_bpb: 1.23641511`
- Delta vs Attempt01:
  - `-0.00173022` bpb (improvement)
- Decision: KEEP
- Evidence status:
  - metric/log paths expected under `evidence/` for attempt02
  - if not yet synced locally, sync before claim-gate review

Interpretation:
- single-delta process is working
- warmdown refinement gives modest but real gain

## Current best

- Best benchmark result so far: `1.23641511` (Attempt02)

## Next planned attempt (Attempt03)

Discipline:
- single delta only
- leaderboard-inspired, low integration risk
- stack on top of Attempt02 (not Attempt01)

Plan:
- Keep `WARMDOWN_ITERS=3500`
- Add one new delta: `EMA`
- Compare only against `1.23641511`

## Claim posture

- Still NON-CLAIMING.
- No leaderboard/SOTA claim until full claim-gate requirements are explicitly passed.
