# Single A100 QAT Performance Fix

## Summary
This non-record submission tunes a standard `modded-nanogpt`-derived parameters stack so that Quantization-Aware Training (QAT) fits robustly within the 10-minute constraint on a single A100. Previous SOTA variants utilized `torch.quantile`, but passing that to Triton generated a severe 30x GPU performance penalty. By pivoting the internal clip factor estimator of `CastedLinear` to `w.abs().amax(dim=1)`, we bypass the compiler issue entirely.

We also constrained the gradient accum sizing from multi-GPU scales down to 131K tokens, ensuring the model successfully clears 2600 descending iterations before gracefully terminating into an SWA and evaluating, instead of starving the LR decay schedule.

## Results
* **Hardware:** 1x A100 (80GB)
* **Training Loop Length:** 10 Minutes (Wallclock Cap - 2600 iterations; excludes final sliding-window evaluation)
* **End-to-End Runtime (Training + Final Sliding-Window Eval):** ~33 Minutes (per `train.log`)
* **Validation BPB:** `1.4078`
* **Artifact Size:** `15.77 MB` (int6 + zstd)

* **Author:** Shuvam Banerji Seal (https://github.com/Shuvam-Banerji-Seal)
