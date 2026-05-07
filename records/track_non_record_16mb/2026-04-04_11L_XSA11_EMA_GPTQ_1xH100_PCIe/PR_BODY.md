## Summary

Non-record / unlimited-compute 16MB submission from a long `1x H100 PCIe` screening run of an 11-layer XSA-all + EMA + legal self-generated GPTQ fork of the `#1019` lineage.

## Result

- Final exact post-quant score: `1.15466807 bpb`
- Final exact post-quant loss: `1.94960867`
- Post-EMA pre-quant score: `1.1416 bpb`
- Total submission size: `15,243,770 bytes`
- Training stop: `4216` steps at `4800.442s`
- Hardware: `1x H100 PCIe 80GB`

## Notes

- This is **not** claiming the 10-minute-on-8xH100 record track.
- The run fits the repo's **Unlimited Compute / Non-record Submissions** lane.
- Started from the merged `#1019` stack and ran from a dedicated branch script.
- GPTQ export was hardened to avoid non-PD Hessian crashes by retrying Cholesky with stronger damping and falling back to percentile int6 quantization if needed.
- An explicit pre-quant checkpoint is saved before export (`final_model_pre_quant.pt`).

## Included

- `README.md`
- `submission.json`
- `train_gpt.py`
- `requirements.txt`

Large local artifacts are intentionally not committed; they are tracked only in `artifacts_manifest.local.json`.
