# Results provenance for `run036-safe016`

These numbers were reconciled from pulled TensorPool artifacts, not copied from bootstrap heuristics.

- **Job ID:** `j-5x7kcly8yl`
- **Canonical bundle:** `~/parameter-golf-project/jobs/run036-safe016/`
- **Pulled artifacts:** `~/parameter-golf-project/state/tp-pulls/run036-safe016/artifacts/`
- **SAFE_SUBMISSION authority:** `final_int6_sliding_window_exact`
- **FRONTIER_ONLY telemetry:** `final_slot_exact`

## Clean-lane authority

| Seed | final_int6_sliding_window_exact | final_int6_roundtrip_exact | total_submission_size_bytes |
|------|--------------------------------:|---------------------------:|----------------------------:|
| 42   | 1.06047528 | 1.07087282 | 15,504,058 |
| 1337 | 1.05812851 | 1.06836276 | 15,457,982 |
| 2024 | 1.05690014 | 1.06682712 | 15,484,283 |
| **Mean** | **1.05850131** | **1.06868757** | **15,482,107.67** |
| **Std (sample)** | **0.00181649** | — | — |

## Frontier-only telemetry (not submission authority)

| Seed | final_slot_exact |
|------|-----------------:|
| 42   | 0.86246414 |
| 1337 | 0.85904408 |
| 2024 | 0.85771266 |
| **Mean** | **0.85974029** |

## Notes

1. `scripts/summarize_tp_pull.py` returned null summary fields for this run because `submission.json` in the pull contains metadata only.
2. Therefore the final `train_seed42.log`, `train_seed1337.log`, and `train_seed2024.log` files are the authoritative source.
3. All three clean-lane artifacts stayed under the 16 MB cap, so no live-log oversize warning survived pull-time reconciliation.
