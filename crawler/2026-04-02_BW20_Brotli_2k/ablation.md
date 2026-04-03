# BW20_Brotli_2k — Ablation Results

Status: pending

## Gate (1k steps, 1-GPU, seed=444)

| Metric | BW5 (zstd) | BW20 (brotli) | Delta |
|--------|-----------|---------------|-------|
| raw_bpb | | | |
| int6_sw_bpb | | | |
| artifact_bytes | | | |
| step_ms | | | |
| compress_time | | | |
| decompress_time | | | |

## Notes
- Model weights identical — only compression backend changed
- Expecting: same BPB, smaller artifact, slightly slower compress/decompress
- Pass criteria: no blowups, roundtrip eval completes, artifact size reduction
