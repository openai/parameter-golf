# Single A100 QAT Performance Fix

## Summary
This non-record submission tunes a standard `modded-nanogpt`-derived parameters stack so that Quantization-Aware Training (QAT) fits robustly within the 10-minute constraint on a single A100. Previous SOTA variants utilized `torch.quantile`, but passing that to Triton generated a severe 30x GPU performance penalty. By pivoting the internal clip factor estimator of `CastedLinear` to `w.abs().amax(dim=1)`, we bypass the compiler issue entirely.

We also constrained the gradient accum sizing from multi-GPU scales down to 131K tokens, ensuring the model can make rapid progress under the 10-minute wallclock cap on a single A100; in the attached `train.log` for this record, training reaches step 1186 before terminating into SWA and evaluation, instead of starving the LR decay schedule.

## Results
* **Hardware:** 1x A100 (80GB)
* **Training Loop Length:** 10 Minutes (Wallclock Cap — run terminates around step 1186 in the attached `train.log`; excludes final sliding-window evaluation)
* **End-to-End Runtime (Training + Final Sliding-Window Eval):** ~33 Minutes (per attached `train.log`)
* **Validation BPB (final sliding-window / roundtrip):** `~1.5252` (per attached `train.log`)
* **Artifact Size:** `15.77 MB` (int6 + zstd)

* **Author:** Shuvam Banerji Seal (https://github.com/Shuvam-Banerji-Seal)
