# Hardik SOTA Run

This submission implements a high-performance configuration designed to push the frontier of the 16MB / 10-minute track in the Parameter Golf challenge.

## Techniques

- **SP8192 Tokenizer**: Uses the 8192-vocab tokenizer for superior compression on the FineWeb dataset compared to the baseline 1024-vocab version.
- **Depth Recurrence (L3-5)**: Layers 3 through 5 are executed twice per forward pass, effectively increasing the model's depth without adding to the parameter count or artifact size.
- **Parallel Residuals**: Processing Attention and MLP in parallel allows for a wider model within the same latency budget, improving representational capacity.
- **Muon Optimizer**: Utilizing the Muon optimizer for matrix parameters, which has shown significant gains in training speed and convergence for constrained runs.
- **Legal Score-First TTT**: Test-time training is applied during evaluation, specifically using the "score-first" approach which is compliant with the challenge rules (only training on tokens already evaluated).
- **GPTQ + SDClip**: Post-training quantization using GPTQ with Standard Deviation Clipping (SDClip) to maximize information density in the 16MB artifact.

## Performance Target

This configuration targets a `val_bpb` of approximately **1.0805**, which would place it at the top of the leaderboard.

## Compliance

- **Training Time**: Optimized to complete in under 600 seconds on 8xH100 GPUs.
- **Artifact Size**: Artifact is managed to stay comfortably under the 16,000,000 byte limit through efficient quantization and Brotli compression of the state dictionary.
- **Reproducibility**: Script is fully self-contained and reproducible across multiple seeds.
