# AWQ 2xH100 Proxy, No-Compile Quantized Eval

Non-record candidate submission by Chandrasen Pandey (`Devchandrasen`).

This is a small-resource reproduction/variant of the PR #1908 / PR #1956 AWQ + GPTQ stack. It was run on 2x H100 instead of the official 8x H100 leaderboard configuration, so it is submitted as non-record evidence rather than a SOTA claim.

## Result

| Metric | Value |
|---|---:|
| Seed | 42 |
| Quantized validation BPB | 1.15828615 |
| Quantized validation loss | 2.53478079 |
| Pre-quant post-EMA BPB | 1.15335094 |
| Training stop | 1241 steps |
| Training time | 599917 ms |
| Quantized artifact bytes | 15964464 |
| Compressed code bytes | 33825 |
| Total counted bytes | 15998289 |
| GPUs | 2x H100 80GB |

The total counted artifact is under the decimal 16MB limit by 1711 bytes.

## Notes

- Based on the PR #1956 record folder, itself a compliant rerun of the PR #1908 activation-aware GPTQ/AWQ stack.
- This run used `LQER_TOP_K=2` to fit under the 16MB cap on the 2-GPU proxy run.
- Test-time training was disabled for this proxy run: `TTT_ENABLED=0`.
- The original full training run produced the under-cap artifact, then crashed during the compiled quantized eval path on the local PyTorch 2.8.0 + CUDA 12.8 HPC environment.
- `train_gpt.py` includes a tiny environment-gated bypass for `torch.compile` in the quantized eval path. The saved under-cap artifact was then reloaded and evaluated successfully with `PGOLF_DISABLE_QUANT_COMPILE=1`.

## Logs

- `train_seed42_original_compile_crash.log` is the full training/serialization log from the 10-minute 2xH100 run. It includes the under-cap artifact size and the original post-decompression compile crash.
- `eval_seed42_existing_artifact_nocompile.log` reloads that same `final_model.int6.ptz` artifact and reports the clean quantized validation score above.

This is not a leaderboard-winning run. It is a packaged non-record/candidate result showing a valid under-cap artifact and clean quantized evaluation on available 2xH100 HPC resources.
