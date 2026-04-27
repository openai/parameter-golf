# CLASE-Quant: Adaptive Layer-Sensitive Quantization + Extended Context

**Mean val_bpb: 1.1914** (3 seeds on 8xH100 SXM)

## Summary

This submission introduces **CLASE-Quant**, an adaptive per-layer quantization strategy inspired by the CLASE Technique (HDXspeed, March 2026), which proved that non-uniform treatment of transformer layers outperforms uniform approaches. Combined with extended 2048 sequence length training, ramping weight decay, and sliding window evaluation, we achieve a significant improvement over the baseline.

## Key Techniques

### 1. CLASE-inspired Adaptive Per-Layer Quantization (Novel)

Not all transformer layers are equal. The CLASE Technique demonstrated that boundary layers (first/last) carry more gradient flow and are more sensitive to perturbation than middle layers, where skip connections provide redundancy.

We apply this insight to post-training quantization:
- **Boundary layers** (blocks 0, 1, N-2, N-1): **int8** quantization (128 levels) — highest precision where sensitivity is highest
- **Middle layers** (blocks 2 through N-3): **int6** quantization (32 levels) — lower precision where skip connections provide error correction
- **Tied embeddings**: **fp16** passthrough — most sensitive due to dual input/output role
- **Control tensors** (scales, gains, resid_mix): **fp32** passthrough — negligible size, high impact

This non-uniform allocation saves ~15% model size compared to uniform int8 while preserving accuracy at the most critical positions.

### 2. Ramping Weight Decay (Novel)

Standard approaches use fixed weight decay throughout training. We increase weight decay from 0.02 to 0.08 during the warmdown phase using a cosine schedule. The intuition: during warmdown, we want weights to converge to tighter distributions that are easier to quantize cleanly. This progressively compresses weight magnitudes, reducing quantization error at export time.

### 3. Extended Context Training (2048 seq len)

Training at 2048 tokens per sequence (up from 1024) provides richer context per training step. Combined with adjusted hyperparameters (lower LR 0.03, higher momentum 0.97, smaller batch 393K tokens), this yields better per-step learning despite seeing fewer total sequences.

### 4. Sliding Window Evaluation (stride=64, seq_len=2048)

Every token is scored with at least 1984 tokens of prior context instead of the 0-2047 average from standard chunked evaluation. This gives a more accurate estimate of model quality, particularly for the extended context we trained with.

### 5. Architecture (from prior work)

Built on proven techniques from earlier submissions:
- **10 transformer layers** (up from 9 baseline)
- **FP16 tied embeddings** with overtone spectral init
- **Phase-transition residual mixing** (sigmoid schedule)
- **Muon optimizer** with decoupled weight decay
- **GQA** (8 heads, 4 KV heads)

## Results

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact |
|------|----------|---------|-------|---------|----------|
| 1337 | 2.01365 | 1.19260 | 10854 | 55.27 | 11.46 MB |
| 42 | 2.00862 | 1.18962 | 12761 | 47.01 | 11.76 MB |
| 7 | 2.01284 | 1.19212 | 13205 | 45.43 | 11.50 MB |
| **Mean** | **2.01170** | **1.19144** | | | |

- All 3 runs completed within 600s wallclock on 8xH100 SXM (RunPod Secure Cloud)
- Peak GPU memory: ~8,500 MiB per GPU
- Artifact size: ~11.5 MB (well under 16 MB limit, 4.5 MB headroom)
- Eval time: ~66-74s (sliding window with compiled forward)

## Improvement over Baseline

| Entry | val_bpb | Delta from Baseline |
|-------|---------|-------------------|
| Baseline | 1.2244 | - |
| **CLASE-Quant (ours)** | **1.1914** | **-0.0330** |

## Quantization Bit Allocation Detail

For 10 layers:
```
Block 0 (first):  int8 (128 levels) - high gradient sensitivity
Block 1:          int8 (128 levels) - near-input sensitivity
Block 2-7:        int6 (32 levels)  - skip connection redundancy
Block 8:          int8 (128 levels) - near-output sensitivity
Block 9 (last):   int8 (128 levels) - high gradient sensitivity
Embeddings:       fp16 passthrough  - dual input/output role
Control tensors:  fp32 passthrough  - negligible size
```

## Acknowledgments

- Built with Claude (Anthropic) as AI pair programmer
- Inspired by the CLASE Technique (HDXspeed, March 2026)
- Builds on techniques from notapplica, Matthew Li, samacqua, Spokane Way, Nan Liu, and Renier Velazco
- Training infrastructure: RunPod (8xH100 SXM Secure Cloud)
