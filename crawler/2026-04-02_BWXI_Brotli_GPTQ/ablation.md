# BW XI — Ablation Results

Status: pending

## Parent: BWX 9F
| Metric | BWX (zstd, no GPTQ) |
|--------|---------------------|
| int6_sw_bpb | 1.13867894 |
| bytes_total | 15,239,617 |
| step_ms | 110.19 |
| steps | 5446 |

## Full Run (8×H100, 600s, seed=444)

| Metric | BW XI (brotli + GPTQ) | Delta vs BWX |
|--------|-----------------------|--------------|
| raw_bpb | | |
| int6_sw_bpb | | |
| bytes_total | | |
| step_ms | | |
| steps | | |
| gptq_cal_s | | |
| artifact_legal | | |

## Confirmation (seed=300)

| Metric | BW XI seed=300 |
|--------|----------------|
| int6_sw_bpb | |
| bytes_total | |

## Notes
- Brotli: approved baseline (BW20 gate clean)
- GPTQ: confirmed −0.002 signal in BW12/BW13
- Expect: ~−0.002 BPB, artifact stays under 16MB thanks to brotli offset
