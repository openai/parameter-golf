# Error Correction Network (ECN) — Novel Research

## Concept

A tiny neural network (~1000 parameters) that corrects the logits of the main model during evaluation, learning per-token from its own prediction errors.

The network learns: "when the model is uncertain AND recent accuracy is dropping, shift probability toward frequent tokens." These are non-linear interactions that simple bias correction cannot capture.

## Results

| Method | BPB Improvement | Speed | Usable in 10min? |
|--------|----------------|-------|-----------------|
| ECN (backprop) | **-0.039** | 28 wps | No |
| ECN freeze-after-warmup | -0.006 | 28 wps | No |
| BCC (binned context correction) | -0.002 | 1107 wps | Borderline |
| Bias correction | 0.000 | 608 wps | No |
| Hybrid bigram | +0.001 (worse) | 1200 wps | No |
| Ridge regression (FEC) | 0.000 | 1075 wps | No |
| Self-refine (2-pass) | +0.003 (worse) | 147 wps | No |

## Key Finding

The ECN achieves **-0.039 BPB** — larger than most TTT implementations in this challenge — at **zero artifact cost**. The correction network is created in code and learns during evaluation. No weights stored.

The fundamental bottleneck is PyTorch autograd overhead on per-token updates, not the mathematics (which is only ~2000 FLOPs per token for 1000 parameters).

## Why It Works

The ECN does NOT predict the next token (the main model already does that better than any simple corrector). Instead it learns the model's **systematic errors**:

- Underprediction of common tokens after punctuation
- Overconfidence on rare tokens in low-entropy contexts
- Calibration drift over long documents

This is fundamentally different from:
- **Hybrid/ensemble approaches** (which failed — mixing with a weaker predictor always hurts)
- **Simple bias correction** (which lacks the non-linear feature interactions)
- **Pre-trained frozen correction** (which can't adapt to the specific validation data)

## Future Work

- Custom Triton/CUDA kernel for fused forward+backward could achieve 100x speedup
- ELM + Top-K RLS (frozen random features + recursive least squares on top-32 logits) as gradient-free alternative — currently in development, numerically challenging but promising
- If eval time limit were extended to 30 minutes, ECN would achieve ~1.028 - 0.039 = **0.989 BPB**

## Additional Research: Adapters on Random Linear Maps

First implementation of "Learning adapters on random linear maps" — an item on OpenAI's README wishlist.

| Model | val_bpb | Size | BPB x MB |
|-------|---------|------|----------|
| Baseline (full weights) | 1.2102 | 24.5 MB | 29.6 |
| Adapter (rank-32, no compile) | 1.5584 | 7.92 MB | 12.3 |
| Shared adapter (20 layers) | 1.5839 | 14.1 MB | 22.3 |

The adapter approach stores a seed (4 bytes) + low-rank correction instead of full weight matrices. The random basis is regenerated from the seed at load time (zero storage cost). Model fits in half the 16MB budget.

Key discovery: `torch.compile` creates a numerical divergence between training and evaluation modes for adapter models. Training in eager mode (no compile) resolves the roundtrip mismatch completely.

## Additional Research: Test-Time Training (TTT)

Extensive TTT experiments on transformer models:

| Model | Standard BPB | TTT BPB | TTT Effect |
|-------|-------------|---------|------------|
| 500-step transformer | 1.6641 | 1.6544 | -0.010 |
| 3000-step transformer | 1.4617 | 1.4744 | +0.013 (worse) |

Key finding: TTT helps weak models but **hurts strong models** — the SGD updates overwrite learned knowledge in well-trained models.

## Research Timeline

All experiments conducted over **2 days** (April 13-14, 2026). Research progression:

1. Local setup (RTX 4070 Laptop, WSL2) → baseline training
2. RunPod A40 → 5000-step strong baseline (1.2102 BPB)
3. RunPod 1xH100 → SOTA reproduction (1.0892 BPB)
4. ECN/BCC/hybrid experiments → discovered 0.039 BPB correction
5. GDN-Hybrid discovery → 1.027 BPB (current submission)
6. RunPod 8xH100 → 3-seed cold-cache verification

## Author

**Hamza Koyuer** ([@Hkoyuer](https://github.com/Hkoyuer))
HBO-ICT, Amsterdam University of Applied Sciences (HvA)
[Helolinks.com](https://helolinks.com)
