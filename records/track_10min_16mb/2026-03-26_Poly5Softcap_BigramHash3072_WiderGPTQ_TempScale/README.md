# Poly5 Softcap + BigramHash(3072) + Wider GPTQ-lite + Temperature Scaling + Z-Loss

**11L 512d + LeakyReLU(0.5)² + Poly5 Softcap + BigramHash(3072) + Wider GPTQ-lite (9-pct) + Z-Loss + Temperature Scaling + Legal TTT + Parallel Muon**

## Summary

This submission builds on the current SOTA (LeakyReLU² + Legal TTT + Parallel Muon, 1.1194 BPB) with 6 targeted improvements, each with evidence from ablations or related submissions:

## Improvements over SOTA

### 1. Poly-5 Softcap (replaces tanh)
- **What:** Degree-5 polynomial approximation of tanh: `x * (1 - x²/3 + x⁴/15)` clamped to [-1, 1]
- **Why:** Better torch.compile kernel fusion (tanh breaks fusion per ternary submission analysis: 16ms/step faster). Also provides smoother gradient landscape.
- **Evidence:** Used successfully in the ternary submission (1.1570 BPB). Critical finding from that work: "Switching to tanh broke fusion — F63 was 16ms/step slower."
- **Expected impact:** ~0.001 BPB from faster training (more steps in 10 min) + marginal quality improvement

### 2. BigramHash(3072) (up from 2048)
- **What:** Increased bigram hash embedding vocabulary from 2048 to 3072
- **Why:** The SOTA's own ablation table shows `BigramHash 2048→3072: -0.0009 BPB`
- **Expected impact:** -0.0009 BPB

### 3. Wider GPTQ-lite Percentile Search (9 candidates vs 5)
- **What:** Expanded quantization clip percentile candidates from `[0.999, 0.9995, 0.9999, 0.99999, 1.0]` to `[0.998, 0.999, 0.9993, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1.0]`
- **Why:** More candidates = lower MSE reconstruction error for each weight row. Zero training cost (post-training only).
- **Expected impact:** -0.0001 to -0.0003 BPB

### 4. Temperature Scaling (T=0.95) at Eval
- **What:** Apply temperature=0.95 to logits during sliding window and TTT evaluation
- **Why:** Sharpening the distribution can improve BPB when the model is slightly oversmoothed. Used in ternary submission (T=0.90).
- **Expected impact:** -0.001 to -0.002 BPB (conservative T=0.95 vs aggressive T=0.90)

### 5. Z-Loss Regularization (weight=1e-4)
- **What:** Added `z_loss = 1e-4 * mean(logsumexp(logits)²)` to training loss
- **Why:** Stabilizes logit magnitudes, prevents loss spikes, improves training stability. Standard technique from PaLM/Gemini.
- **Expected impact:** -0.0005 to -0.001 BPB from more stable training

### 6. LZMA Preset 9 Compression (up from 6)
- **What:** Increased lzma compression level from 6 to 9 (maximum)
- **Why:** Better compression ratio means smaller artifact, leaving more headroom for parameters
- **Expected impact:** ~0.5-1% smaller artifact, marginal quality improvement

## Architecture (preserved from SOTA)

- **Layers:** 11 (512d, 8 heads, 4 KV heads GQA)
- **MLP:** 3x expansion, LeakyReLU(0.5)² activation
- **Attention:** XSA on last 4 layers, Partial RoPE (16/64 dims)
- **Embeddings:** BigramHash(3072, 128d), tied embeddings, ValueEmbedding(128d, layers 9-10)
- **Normalization:** RMSNorm with LN Scale (1/sqrt(layer+1))
- **U-Net:** 5 encoder + 6 decoder with learned skip weights
- **SmearGate:** Token blending with learned gate

## Training (preserved from SOTA)

- **Optimizer:** Parallel Muon (batched Newton-Schulz) for banks + AdamW for embeddings/scalars
- **LR:** matrix=0.025, scalar=0.025, tied_embed=0.035
- **Momentum:** 0.99 (warmup from 0.92 over 1500 steps)
- **Weight decay:** 0.04 (both Muon and Adam)
- **Warmdown:** 3500 iterations
- **Weight averaging:** EMA(0.997) + Tight SWA (every 50 steps when scale < 0.2)
- **Late QAT:** STE int6 when LR scale < 0.15

## Evaluation

- **Sliding window:** stride=64 with temperature scaling (T=0.95)
- **Legal TTT:** Score-first protocol, 3 epochs SGD per chunk, all blocks unfrozen
- **Quantization:** GPTQ-lite int6 (9-percentile search) + lzma-9 compression

## Run Command

```bash
torchrun --nproc_per_node=8 train_gpt.py
```

With TTT enabled:
```bash
TTT_ENABLED=1 torchrun --nproc_per_node=8 train_gpt.py
```

## Expected Results

Based on individual technique deltas:
- SOTA baseline: 1.1194 BPB
- + Poly5 softcap: ~-0.001 (faster steps + quality)
- + BigramHash 3072: ~-0.0009 (ablation-proven)
- + Wider GPTQ-lite: ~-0.0002 (better quantization)
- + Temperature T=0.95: ~-0.001 (conservative estimate)
- + Z-loss: ~-0.0005 (training stability)
- + LZMA-9: marginal
- **Estimated total: ~1.116 BPB** (before TTT), **~1.113 BPB** (with TTT)

Note: Actual results require 8xH100 training. Individual improvements may not stack linearly.
