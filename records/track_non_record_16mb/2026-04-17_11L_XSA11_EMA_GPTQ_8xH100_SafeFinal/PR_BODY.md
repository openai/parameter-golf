## Summary

Non-record `16MB` submission for a legal `8x H100 / 600s` run of an `11L XSA-all + EMA + legal self-generated GPTQ` stack.

## Result

- Final exact sliding-window score: `1.11355040 bpb`
- Final exact sliding-window loss: `1.88017824`
- Final exact post-quant roundtrip score: `1.13694354 bpb`
- Total submission size: `15,353,950 bytes`
- Training stop: `6460` steps at `600.119s`
- Hardware: `8x H100 80GB HBM3`

## Notes

- Legal `8x H100 / 600s` run under the `16MB` cap.
- Uses the official RunPod Parameter Golf template.
- GPTQ calibration is autoregressive self-generated data only.
- Large local artifacts are not committed; checksums are recorded in `artifacts_manifest.local.json`.

## Included

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed1337.log`
- `requirements.txt`
