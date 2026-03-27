# Entropy-Regularized QAT (WIP - pending compute)

Built on the merged 1.1233 record (PR #374). Adds compressibility-aware training via entropy penalty on quantized weight symbols during late QAT window.

Key changes from baseline:
- LeakyReLU(0.5) activation (-0.003 BPB proven)
- Differentiable soft_code_entropy() on int6 quantized MLP weights
- Entropy penalty gated by ENTROPY_LAMBDA env var (default 0, no behavior change)
- Computed every 25 steps during late QAT (LR scale < 0.15)

Status: Code complete, tested locally. Awaiting 8H100 runs for results.
