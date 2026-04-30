# PR 352 — Memory Tokens + Mixed Quantization

**Author:** Austin Tarango (sp00mm)
**Branch created:** 2026-03-21
**Claimed BPB:** 1.16593 (sliding window stride=128, post int5/int6+zstd roundtrip)
**Artifact size:** 15,070,662 bytes
**Seeds:** not stated (single run)

## Files retrieved
- `records__track_10min_16mb__2026-03-21_MemoryTokens__README.md`
- `records__track_10min_16mb__2026-03-21_MemoryTokens__submission.json`
- `records__track_10min_16mb__2026-03-21_MemoryTokens__train_gpt.py`

## Claimed changes (from README, verbatim)

> ## Novel Contribution: Memory Tokens
> 64 learnable embedding vectors that overwrite the first K positions of every input sequence. All real tokens can attend to them via the causal mask, giving every position access to learned global context — a shared scratchpad that the model optimizes end-to-end.
> - Cost: 32,768 parameters (0.12% of model), zero compute overhead
> - A/B tested: -0.014 BPB improvement vs identical config without memory tokens (1.2787 vs 1.2928 sliding, 1xH100)
> - Implementation: Memory positions use ignore_index=-100 so they contribute zero to loss. During sliding window eval, memory tokens are prepended (not overwritten) to preserve all real token context
> - Memory tokens are exempt from weight decay — they're a learned scratchpad that needs to hold specific values, not be regularized toward zero
>
> ## Architecture
> - 10 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA)
> - 3x MLP expansion (hidden=1536), relu^2 activation
> - U-Net skip connections, tied embeddings
> - Memory tokens (64): global context scratchpad prepended to every sequence
> - BigramHashEmbedding (10240): hash consecutive token pairs for local context
> - SmearGate: learned blend with previous token at embedding level
> - Partial RoPE (16/64 dims): position encoding on 25% of head dims, rest is content-only
> - LN Scale: RMSNorm output damped by 1/sqrt(layer+1) for stability
>
> ## Training
> - Muon optimizer (matrix_lr=0.04, momentum=0.95) + AdamW (embed/scalar, WD=0.04)
> - Muon weight decay (0.04), memory tokens exempt from WD
> - MTP auxiliary heads (k=2, alpha=0.2, stripped before export)
> - EMA (decay=0.997, on-device, every 10 steps)
> - Late QAT: fake int6 quantization (STE) when lr_scale < 0.1
> - seq_len=2048, batch=524K tokens, warmdown=3000, grad_clip=0.3
> - 9,030 steps in 600s (64ms/step)
>
> ## Quantization
> - Int5 [-16,15] for MLP weights (most compressible)
> - Int6 [-32,31] for attention weights (precision-sensitive)
> - FP16 for tied embeddings and small tensors
> - zstd-22 compression (better ratio than zlib)

Pre-quant 1.1842; int6+zstd roundtrip 1.1820; sliding window 1.1659. Model params 25,812,049.
