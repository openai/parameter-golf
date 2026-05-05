# Experiment 088: 10L + INT5_MLP QAT + leaky2 + 8×H100 — INT5 QAT HURTS!

## Config
- 10 layers, model_dim=512, MLP_HIDDEN=1536
- leaky_relu(0.5)² activation
- INT5_MLP=1 with int5 QAT during training (fake quantize MLP to [-16,15])
- No FP16_KEEP, No BigramHash
- NorMuon + QAT (int5 for MLP, int6 for attn), WD=0.04, LR=0.025
- 8×H100 via torchrun, ~90.67ms/step

## Results
- Steps: 6,618 @ 90.67ms/step
- **Standard BPP: 1.1679** (WORSE than 087's 1.1602 by 0.008!)
- **Sliding BPP: 1.1468** (WORSE than 087's 1.1391 by 0.008!)
- Manual+zstd: 16.72MB ❌ (BIGGER than 087's 16.52MB!)
- FLAT+zstd: **16.61MB ❌** (BIGGER than 087's 16.25MB!)

## CRITICAL FINDING: INT5 QAT HURTS EVERYTHING
- BPP: 0.008 worse (1.1468 vs 1.1391)
- Artifact: 360KB bigger (16.61MB vs 16.25MB)
- Int5 fake quantization during training constrains MLP weights too aggressively
- The model loses capacity from narrower quantization range during optimization

## Root Cause
Andrewgcodes uses int6 QAT for ALL layers during training. Int5 is ONLY applied at
post-training quantization for MLP weights. This means:
- During training: full int6 range [-32,31] for all weights → better model quality
- At serialization: int5 range [-16,15] for MLP → better compression
- The training doesn't "know" about int5, so it optimizes freely

## Fix for exp089
Keep QAT=int6 for all layers during training (same as exp087).
Add INT5_MLP=1 for post-training quantization only.
Plus TTT + pruning for extra BPP + compression gains.

## Comparison
| Exp | INT5 QAT | INT5 post-quant | Sliding BPP | FLAT+zstd |
|-----|----------|-----------------|-------------|-----------|
| 087 | no | no | 1.1391 | 16.25MB |
| **088** | **yes** | **yes** | **1.1468** | **16.61MB** |
| 089 (planned) | no | yes | ~1.1391? | ~14.5MB? |
