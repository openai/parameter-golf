# PR 535 — Record: 11L LeakyReLU² + Full GPTQ + QAT Alignment (val_bpb: 1.1204)

**Author:** Raahil Shah (raahilshah)
**Claimed BPB:** val_bpb 1.12037520 (val_loss 1.89170159), 3-seed mean (std 0.00011489). Pre-quant val_bpb 1.1374 / val_loss 1.9210.
**Artifact size:** 15,851,228 bytes max across seeds
**Seeds:** 7 (1.12026948, 15,762,694 bytes), 314 (1.12049747, 15,732,473 bytes), 2024 (1.12035866, 15,851,228 bytes)

## Files retrieved
- `records__track_10min_16mb__2026-03-23_11L_LeakyReLU_GPTQ_QATalign__README.md`
- `records__track_10min_16mb__2026-03-23_11L_LeakyReLU_GPTQ_QATalign__submission.json`
- `records__track_10min_16mb__2026-03-23_11L_LeakyReLU_GPTQ_QATalign__train_gpt.py`

## Environment variables (from run command in README)

```
SEED=7 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Claimed changes (from README, verbatim)

> val_bpb: 1.1204 (3-seed mean, std 0.0001) | 15.85 MB max artifact | 8xH100 SXM, 600s.
>
> Key Innovations:
>
> 1. LeakyReLU(0.5)² Activation. Replace relu(x)² with leaky_relu(x, 0.5)² in the MLP. Standard relu² zeroes out negative activations, creating dead neurons whose down-projection weights are wasted capacity. LeakyReLU(0.5)² maps negatives to (0.5x)² = 0.25x², allowing the down projection to see non-zero gradients from all neurons. This effectively doubles MLP capacity without adding parameters. Observable as dramatically faster early training convergence.
>
> 2. Full GPTQ Quantization. Hessian-aware GPTQ replaces percentile-search quantization. For each weight matrix: 256-sample calibration set from training data; per-layer Hessian approximation (H = X^T X); column-wise int6 quantization with Cholesky error compensation; block size 128, column reordering by ascending Hessian diagonal. Reduces int6 quantization gap from 0.0085 to 0.0059 BPB (31% reduction).
>
> 3. QAT-Export Alignment. The STE fake-quantizer during training must match the export quantizer. We use quantile(0.9995) for per-row clipping in both the STE and the final export path, ensuring training optimizes against the actual quantization that will be applied.
>
> Architecture: 11 transformer layers, dim=512, 8 heads, 4 KV heads (GQA); 3x MLP (1536) with LeakyReLU(0.5)² activation; XSA on last 4 layers; Partial RoPE (16/64 dims) + NTK-aware scaling; LN Scale 1/sqrt(layer_idx+1); U-Net skip connections (5 encoder, 6 decoder); SmearGate; BigramHash (2048, 128-dim); Shared VE (dim=128, layers 9-10); FA3; orthogonal init, logit softcap 30, tied embeddings.
>
> Training: Muon (matrices) lr=0.025 momentum=0.99 WD=0.04; AdamW (emb) lr=0.035, (scalars) lr=0.025, WD=0.04; grad clip 0.3; batch 786432 tokens/step seq_len=2048; Warmdown 3500 iters (wallclock-based); EMA decay=0.997 + Tight SWA (every 50 steps, scale<0.2); Late QAT STE int6 when LR scale<0.15.
>
> Quantization: Full GPTQ with 256-sample Hessian calibration; Int6 per-row with quantile(0.9995) clipping; small tensors + tok_emb.weight in fp16; zstd level 22.
>
> Per-seed: 7 → ~6820 steps, 1.8915, 1.1203, 15,762,694 bytes. 314 → ~6820, 1.8919, 1.1205, 15,732,473 bytes. 2024 → ~6820, 1.8917, 1.1204, 15,851,228 bytes. Mean 1.1204, std 0.0001.
