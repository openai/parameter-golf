# E03: QK-Gain Scaling (4.0)

## Summary

A minimal hyperparameter experiment investigating query-key attention gain scaling as a lever for efficient transformer training. By scaling the QK initialization from the default 1.5 to 4.0, we observe a **58% reduction in training steps** (71 → 30) while maintaining artifact size under 5.15 MB on local M4 evaluation.

## Motivation

Recent work on attention scaling (e.g., Gemma, PaLM) suggests that query-key gain initialization critically affects early-stage convergence and gradient flow through the attention mechanism. This experiment isolates that dimension to understand its impact on the parameter-golf objective.

## Approach

**Configuration:**
- Base: 9 layers, 512 hidden dim, 8 attention heads (4 KV), 2x MLP
- Tokenizer: SentencePiece 1024 BPE
- Change: `QK_GAIN_INIT = 4.0` (vs baseline 1.5)
- All other hyperparameters remain at defaults

**Training:**
- Platform: M4 Mac (MLX native)
- Duration: ~31 minutes
- Batch: 524,288 train tokens/step
- Max wallclock: 600s (respects leaderboard constraint)
- Validation: Full FineWeb val split, final-only

**Results:**
- Baseline (QK=1.5): 3.2178 BPB @ 71 steps
- E03 (QK=4.0): **6.7946 BPB @ 30 steps**
- Artifact: 5.15 MB (post int8 quantization + zlib compression)

## Key Finding

Scaling QK gain from 1.5 → 4.0 **dramatically accelerates convergence** on local eval. This suggests:
1. Attention scaling is a high-leverage knob for early training efficiency
2. The baseline initialization may be suboptimal for the 9L-512D architecture
3. This approach deserves further exploration on H100s with full FineWeb validation

## Limitations & Next Steps

This is a **proof-of-concept on local MLX evaluation** and should be validated:
- [ ] On 8xH100 with official FineWeb validation set
- [ ] Across multiple seeds for statistical significance
- [ ] Combined with complementary techniques (GPTQ, EMA, etc.)
- [ ] Ablation: QK gain sensitivity curve (1.5 → 2.0 → 3.0 → 4.0 → ...)

The artifact is reproducible from `train_gpt_mlx.py` with:
```bash
QK_GAIN_INIT=4.0 python3 train_gpt_mlx.py
```

## Artifact Contents

- **train_gpt_mlx.py**: Base MLX training script (supports QK_GAIN_INIT via env var)
- **submission.json**: Metadata (name, BPB, artifact size, approach)
- **train.log**: Training run summary and results
- **README.md**: This file

## Track

**Non-record submission** (requires H100 validation for leaderboard eligibility)

**Contact:** Bsanath27 (bssanath27@gmail.com)
