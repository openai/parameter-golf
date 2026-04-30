# Autoresearch-Guided Hyperparameter Optimization + SLOT + MuonEq-R

## Summary

Systematic hyperparameter exploration via autoresearch methodology: 100+ automated experiments (80+ on Mac, 10 on H100) to validate training improvements, combined with SLOT logit-bias test-time adaptation and MuonEq-R optimizer.

**Non-record submission**: Documenting findings and negative results from systematic search.

## Key Findings

### What Works (validated on H100)
- **QK-Gain=3.0**: -0.002 BPB improvement vs default 1.5 (validated on 1xH100, 5-min runs)

### What Doesn't Work
- **QK-Gain=5.0**: +0.035 BPB worse (contradicts some PR claims)
- **Matrix LR=0.03 or 0.05**: Both worse than default 0.04
- **Muon momentum=0.95**: Worse than default 0.99
- **Warmdown=4000**: Much worse than default 3500
- **Batch 1M tokens**: Worse (fewer steps)
- **MuonEq-R with Parallel Muon**: Row-normalization interacts badly with banked/sharded weight format, causing ~40% regression
- **Int4 quantization**: 25x worse MSE than int6, not viable
- **Hadamard rotation**: Only 1.16x improvement, not worth the complexity
- **init_std=0.02**: Helps with AdamW but hurts with Muon optimizer

### H100 Autoresearch Results (baseline train_gpt.py, 1xH100, 5 min)

| Config | val_bpb | vs Baseline |
|---|---|---|
| QK-Gain=3.0 | 1.4192 | -0.0024 |
| Baseline (default) | 1.4216 | — |
| Momentum=0.95 | 1.4343 | +0.013 |
| MatrixLR=0.05 | 1.4371 | +0.015 |
| MuonWD=0.08 | 1.4471 | +0.026 |
| QK-Gain=5.0 | 1.4562 | +0.035 |
| MatrixLR=0.03 | 1.4694 | +0.048 |
| Batch=1M | 1.4758 | +0.054 |
| Warmdown=4000 | 1.5640 | +0.142 |

### Mac Autoresearch Results (80+ experiments, simplified model, AdamW)

| Finding | Effect | Transfers to H100? |
|---|---|---|
| init_std=0.05 (vs 0.005) | -8.8% val loss | NO (Muon optimizer makes init less important) |
| Wider models (640d) | -1.3% val loss | Partially |
| softcap=15 (vs 30) | -0.5% val loss | Unknown |
| Larger batch (grad_accum=4) | -5.7% val loss | N/A (already large on H100) |
| GELU, SwiGLU activations | Worse | Yes (confirmed: relu_sq is best) |

## Methodology

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): automated experiment loops that modify hyperparameters, train, evaluate, and keep/discard changes.

### Novel Techniques Explored (negative results)

1. **HadaGPTQ**: Hadamard rotation before GPTQ quantization (inspired by PolarQuant, QuIP#, TurboQuant). Only 1.16x MSE improvement — not enough to justify complexity.

2. **Int4 GPTQ + larger model**: Quantize to 4 bits to fit 45M params in 16MB. QAT at int4 causes +27.7% val loss degradation — not viable.

3. **MuonEq-R with Parallel Muon**: Row-normalizing momentum in the banked optimizer causes training regression when weights are sharded across GPUs.

## Reproduction

```bash
SEED=42 QK_GAIN_INIT=3.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
