# BW XI — Ablation Results

Status: pending

## Parent: BWX 9F
| Metric | BWX (zstd, no GPTQ, loops=3, QK1.5) |
|--------|--------------------------------------|
| int6_sw_bpb | 1.13867894 |
| bytes_total | 15,239,617 |
| step_ms | 110.19 |
| steps | 5446 |

## Changes stacked
1. Brotli compression (approved)
2. Loop-aware GPTQ (confirmed −0.00380 in BW10)
3. QK_GAIN_INIT=4.0 (high-confidence, first crawler test)
4. CRAWLER_LOOPS=2 (directional −0.054 in BW17 RAPID)

## Full Run (8×H100, 600s, seed=444)

| Metric | BW XI | Delta vs BWX |
|--------|-------|--------------|
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
- If loops=2 hurts: rerun with CRAWLER_LOOPS=3 (isolate)
- If QK4 hurts: rerun with QK_GAIN_INIT=1.5 (isolate)
- If artifact > 16MB: check brotli vs zstd delta, consider GPTQ_CAL_SAMPLES=64
