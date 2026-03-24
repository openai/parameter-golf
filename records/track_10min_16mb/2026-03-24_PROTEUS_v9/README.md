# PROTEUS v9 — Parameter Golf Submission

**Built with [PROTEUS](https://lightspeedup.com) by LightSpeedUp**

## Result

**Mean val_bpb: 1.1526** (3 seeds, std: 0.0004)

| Seed | Post-Quant BPB | TTT BPB (1 epoch) | Steps | Step Avg |
|------|---------------|-------------------|-------|----------|
| 42   | 1.1804        | 1.1527            | 6989  | 85.7ms   |
| 1337 | 1.1749        | 1.1529            | 6997  | 85.8ms   |
| 2024 | 1.1771        | 1.1522            | 7093  | 84.6ms   |

## TTT Legality — Single Epoch, Score-Then-Train

This submission uses **single-epoch TTT** (`TTT_EPOCHS=1`), addressing the ruling on PR #568 ([comment](https://github.com/openai/parameter-golf/pull/568#issuecomment-4119903415)) where multi-epoch TTT was correctly identified as training on eval data.

**Why single-epoch is different:**

In single-epoch, each chunk is processed left-to-right within the document:
1. Forward pass on chunk → **score** (accumulate loss for BPB)
2. **Train** LoRA on that chunk's loss (backward-looking)
3. Advance to next chunk

Each token is scored **exactly once**, **before** being trained on. The LoRA adapts to the document's distribution but never scores tokens it has already trained on. This is the same score-then-train pattern as PR #77 (merged), applied once per document.

**What changed from v7/v8:** `TTT_EPOCHS` reduced from 2-5 to 1. No other code changes.

## Architecture

- 11 transformer layers, dim=512, 8 heads / 4 KV heads (GQA)
- MLP 3x expansion (1536 hidden), relu² activation
- SmearGate + BigramHash(2048, dim=128) + OrthoInit
- Depth-scaled residual: `1/sqrt(layer_idx + 1)` per block
- U-Net skip connections, tied embeddings
- RoPE base 50K with NTK-aware eval scaling
- XSA on last 4 layers
- 26.8M parameters

## Training

- Muon optimizer (matrix_lr=0.025, WD=0.04, momentum=0.99)
- AdamW for embeddings/scalars (WD=0.04)
- Batch size: 786,432 tokens
- Warmdown: 3000 iterations, wallclock-based
- EMA 0.997 every step
- 3% magnitude pruning before export
- Gradient clipping: 0.3

## Quantization

- **INT6 GPTQ-lite** — 5 clip percentiles per row, pick lowest MSE
- FP16 for tied embeddings
- FP32 for control tensors (scales, mixes, gains)
- zstd-22 compression
- Artifact: ~15.4 MB (96.3% of 16MB budget)
- Quant gap: 0.006 BPB

## Test-Time Training (TTT)

- **Single epoch** — each token scored once before training
- LoRA rank 8, targets: Q + V projections + LM head
- Optimizer: Adam (lr=0.01, betas 0.9/0.95)
- Batch: 64 documents (independent LoRA per document)
- Min document length: 512 tokens
- Eval time: ~110-115s (within 600s budget)
- TTT gain: ~0.025 BPB over post-quant baseline

## Platform

Trained on RunPod 8xH100 SXM, PyTorch 2.8.0+cu128.
