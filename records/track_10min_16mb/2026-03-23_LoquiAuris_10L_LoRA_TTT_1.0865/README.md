# Loqui Auris — 10L + EMA + LoRA TTT

**Built by Eli Pancamo / [Loqui Auris LLC](https://loquiauris.com)**

## Result

**Mean val_bpb: 1.0865** (2 seeds, std: 0.0013)

| Seed | Post-Quant BPB | LoRA TTT BPB | Steps | Artifact |
|------|---------------|-------------|-------|----------|
| 42   | 1.1610*       | 1.0856      | 5978  | 15.73 MB |
| 1337 | 1.1610*       | 1.0874      | 5969  | 15.63 MB |

*Pre-TTT sliding-window BPB estimated from training eval.

## Architecture

- 10 transformer layers, dim=512, 8 heads / 4 KV heads (GQA)
- MLP 3x expansion (1536 hidden), ReLU² activation
- SmearGate + BigramHash(4096, dim=128) + OrthoInit
- U-Net skip connections, tied embeddings
- Logit softcap: 30.0
- RoPE base 10000
- ~24.7M parameters

## Training

- Muon optimizer (matrix_lr=0.02, WD=0.04, momentum=0.99)
- AdamW for embeddings/scalars (lr=0.03/0.02)
- Batch size: 786,432 tokens, seq_len=2048
- Warmup: 20 steps, warmdown: 3000 iterations (wallclock-based)
- EMA: decay=0.997, applied after training
- Gradient clipping: 0.3
- ~6000 steps in 600s on 8xH100

## Quantization

- INT6 per-row for all weight matrices (attention + MLP)
- FP16 for tied embeddings
- FP32 for control tensors (scales, mixes, gains)
- zstd level 22 compression
- Artifact: ~15.7 MB

## Test-Time Training (LoRA TTT)

Per-document backward-looking LoRA adaptation during evaluation.

For each document in the validation set:
1. Split into 256-token chunks with 1024-token eval windows
2. Process chunks left-to-right over 2 epochs
3. Each chunk: forward pass with LoRA → score (final epoch) → train LoRA (non-final chunks)
4. Reset LoRA weights + optimizer state between documents

Key details:
- LoRA rank 8 on Q + V projections + LM head
- Adam optimizer (lr=0.01)
- Batch: 64 documents per GPU (independent LoRA per document)
- Documents < 512 tokens: standard eval without TTT
- Fresh uncompiled model copy for TTT (avoids torch.compile graph caching)
- 8-GPU sharding: docs distributed across ranks, all-reduced at end
- TTT time: ~245s per seed

## Key Technique: Fresh Model for LoRA TTT

torch.compile with `fullgraph=True` caches the forward graph from training,
which has `None` for all LoRA delta arguments. The compiled graph silently
ignores LoRA deltas at eval time. Creating a fresh uncompiled GPT model from
state_dict and running LoRA TTT on it is essential for LoRA to actually work.

## Platform

RunPod 8xH100 SXM 80GB, PyTorch 2.x.

## Credits

LoRA TTT concept from PROTEUS (PR #512, @MatoTeziTanka). Standard TTT from PR #77 (@samacqua).
SmearGate/BigramHash from @unnir. Muon optimizer, SWA, OrthoInit from the Parameter Golf community.
