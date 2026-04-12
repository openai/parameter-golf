# Results provenance for `run037-safe017`

These numbers were reconciled from pulled TensorPool artifacts, not from live bootstrap telemetry.

- **Job ID:** `j-jvquftkrwd`
- **Canonical bundle:** `~/parameter-golf-project/jobs/run037-safe017/`
- **Pulled artifacts:** `~/parameter-golf-project/state/tp-pulls/run037-safe017/artifacts/`
- **SAFE_SUBMISSION authority:** `quantized_bpb`
- **Lane:** `SAFE_SUBMISSION` (Track-A / fixed predictor / no-TTT)

## Authoritative pulled metrics

| Seed | EMA BPB | Quantized BPB | XSA BPB | Artifact bytes |
|------|--------:|--------------:|--------:|---------------:|
| 42 | 1.017723 | 1.026791 | 1.031731 | 15,313,984 |
| 1337 | 1.007375 | 1.016586 | 1.020691 | 15,830,308 |
| 2024 | 1.008736 | 1.017995 | 1.023138 | 15,820,201 |
| **Mean** | **1.011278** | **1.02045733** | **1.025187** | **15,654,831.00** |
| **Std (sample)** | — | **0.00553017** | — | — |

## Notes

1. `submission.json` in the pulled artifact directory already carries the correct authoritative `quantized_bpb` mean (1.02045733) and artifact sizes, so unlike `run036-safe016` there is no summary-script blind spot here.
2. The upstream cold-cache claim for PR #1545 was `1.02830800 BPB`; this Joshua-owned TensorPool reproduction landed materially stronger at **1.02045733 BPB**, consistent with the warm-cache / more-steps regime seen in the pulled logs.
3. All three artifacts stayed under the 16,000,000-byte cap, so the candidate remains launch-clean and directly submittable.
