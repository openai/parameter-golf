# Results provenance for `run039-safe019`

These numbers were reconciled from pulled TensorPool artifacts, not from live bootstrap telemetry.

- **Job ID:** `j-xpv7d0b38j`
- **Canonical bundle:** `~/parameter-golf-project/jobs/run039-safe019/`
- **Pulled artifacts:** `~/parameter-golf-project/state/tp-pulls/run039-safe019/artifacts/`
- **SAFE_SUBMISSION authority:** `quantized_bpb`
- **Lane:** `SAFE_SUBMISSION` (Track-A / fixed predictor / no-TTT)

## Authoritative pulled metrics

| Seed | EMA BPB | Quantized BPB | XSA BPB | Artifact bytes |
|------|--------:|--------------:|--------:|---------------:|
| 314 | 1.007670 | 1.016476 | 1.020950 | 15,522,111 |
| 777 | 1.007192 | 1.016192 | 1.020919 | 15,814,260 |
| 2718 | 1.009535 | 1.018633 | 1.023874 | 15,981,262 |
| **Mean** | **1.008132** | **1.01710033** | **1.021914** | **15,772,544.33** |
| **Std (sample)** | — | **0.00133490** | — | — |

## Notes

1. `submission.json` in the pulled artifact directory carries the authoritative `quantized_bpb` mean (1.01710033) and artifact sizes for this cold-cache confirmation run.
2. Compared with Joshua's prior staged 3-seed GDN-Hybrid SAFE_SUBMISSION artifact `run037-safe017` at `1.02045733 BPB`, `run039-safe019` improves the mean by **0.00335700 BPB**.
3. All three artifacts stayed under the 16,000,000-byte cap, so this candidate is clean-lane submittable.
