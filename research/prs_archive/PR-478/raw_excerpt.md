# PR 478 — Record: 11L XSA-all + GPTQ-lite + EMA + Late QAT (val_bpb: 1.12676, 3-seed mean)

**Author:** gowtham0992
**Claimed BPB:** val_bpb 1.12676 (val_loss 1.90248), 3-seed mean (std 0.00034)
**Artifact size:** 15,713,594 bytes (~15.7 MB mean)
**Seeds:** 42, 1337, 2024

## Files retrieved
- `records__track_10min_16mb__2026-03-22_11L_XSA-all_GPTQ-lite_EMA_LateQAT_1.1271__README.md`
- `records__track_10min_16mb__2026-03-22_11L_XSA-all_GPTQ-lite_EMA_LateQAT_1.1271__submission.json`
- `records__track_10min_16mb__2026-03-22_11L_XSA-all_GPTQ-lite_EMA_LateQAT_1.1271__train_gpt.py`

## Environment variables (from run command in README)

```
BACKOUT_ENABLED=0 MAX_WALLCLOCK_SECONDS=600 RUN_ID=v34 SEED=42
python3 -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

## Claimed changes (from README, verbatim)

> val_bpb 1.12676 (3-seed mean, sliding window stride=64) | ~15.7 MB | 8xH100 SXM, 600s.
>
> Key Innovations:
>
> | Feature | Description | Impact |
> | XSA-all(11) | Exclusive Self Attention on ALL 11 layers, not just last 4 | -0.002 BPB vs XSA(4) |
> | GPTQ-lite | 5 clip percentiles per row, pick min MSE reconstruction | -0.0006 BPB (zero training cost) |
> | EMA(0.997) | Exponential moving average every step | Smoother weights, better compression |
> | Late QAT@0.15 | Int6 STE fake-quantization when LR scale < 0.15 | Minimal quant gap without training noise |
> | Raw Binary + zstd22 | Custom binary serialization, no torch.save overhead | ~300KB savings vs torch.save |
> | No Pruning | Int6-all fits without magnitude or row pruning | No quality loss from pruning |
>
> XSA-all: Our Unique Contribution. Standard XSA applies only to the last few layers. We found that applying XSA to ALL 11 layers provides a consistent 0.002 BPB improvement. Early layers benefit from XSA by being forced to encode novel contextual information rather than repeating self-value patterns. XSA-all(11): 1.12676 @ 6764 steps 88.7ms. XSA(4): 1.13266 @ 6998 steps 85.7ms. Despite XSA-all being ~3ms/step slower, the quality gain outweighs the ~230 fewer training steps.
>
> Ablation: Backout Removal. Removing the Backout mechanism (which subtracts middle-layer output) improved results by 0.0035 BPB. With LN Scale + XSA-all already managing information flow, Backout was redundant. With Backout 1.1306 → Without Backout 1.1271.
>
> Architecture: 11 transformer layers, 512-dim, 8 heads (4 KV, GQA); 3x MLP (1536) relu²; U-Net skips (5 enc, 6 dec); XSA on ALL 11 layers (GQA-aware, zero-alloc); Partial RoPE (16/64 dims) NTK-aware; LN Scale 1/√(layer_idx+1); Shared VE (dim=128, layers 9,10) per-layer learned scales; SmearGate + BigramHash (2048, dim=128); tied embeddings, logit softcap 30.0.
>
> Training: FA3 (fallback SDPA); Muon lr=0.025 momentum=0.99 (warmup 0.92→0.99 over 1500) WD=0.04; AdamW emb lr=0.035 scalars lr=0.025 WD=0.04; grad clip 0.3; batch 786,432 tokens seq_len=2048; Warmdown 3500 iters wallclock-based; EMA 0.997 every step (before quant); Tight SWA every 50 steps scale<0.2; Late QAT Int6 STE when LR scale<0.15 (~step 6242); OrthoInit + muP-scaled output projections; 20-step warmup with state reset.
>
> Quantization: GPTQ-lite per-row optimal clip percentile search (5 candidates: 0.999, 0.9995, 0.9999, 0.99999, 1.0) for int6; Int6 per-row for ALL large weights (MLP + attention + bigram + VE); Int8 per-row embeddings; control tensors fp32; Raw binary serialization + zstd level 22 (no torch.save overhead).
>
> Per-seed: 42 → 6764 steps 1.12713 15.64 MB 88.7ms. 1337 → 6766 1.12648 15.62 MB 88.7ms. 2024 → 6764 1.12667 15.88 MB 88.7ms. Mean 6765, 1.12676, ~15.7 MB, 88.7ms. std 0.00034.
