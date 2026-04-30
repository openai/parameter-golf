# 12L sp9000 — val_bpb 1.0902

**val_bpb: 1.0902** (sliding) | **Artifact: 14.96 MB** | 8×H100 SXM 80GB | 600s

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.0936 |
| Post-quant val_bpb | 1.1066 |
| **Sliding window (stride=64)** | **1.0902** |
| Training steps | 4,544 |
| Step time | 132.0 ms |
| Total training time | 600s |
| Sliding eval time | 155s |
| Artifact size | 14,962,936 bytes |
| Parameters | 36,100,333 |
| Seed | 1337 |

## What Changed vs My Prior Submission

I dropped val_bpb from **1.1036 → 1.0902** (−0.0134) by adding four small training-side levers on top of the previous 12L sp9000 baseline:

| Lever | Mechanism | Param cost |
|---|---|---|
| **AttnOutGate** | Per-head sigmoid gate on attention output (`2.0 * sigmoid(W @ x[:12])`), zero-init weights | 8 heads × 12 layers = 96 params |
| **SmearGate** | Residual stream pre-attention mixer (`x = x + λ * sigmoid(W @ x[:12]) * prev_x`), zero-init weights | 13 params |
| `enable_looping_at=0.45` | Later loop activation (was 0.37) — gives more pre-loop training time, sharper post-loop alignment | 0 |
| `matrix_lr=0.026` | Slightly higher Muon matrix LR (was 0.022) — paired with later loop activation | 0 |

Both gates (~109 params total) are zero-initialized so they're transparent at training start; the model learns to selectively gate per-head attention output and mix prior-token content into the residual stream.

I also dropped TTT for this submission — TTT didn't beat sliding on this stack (both 1.0902), so I removed it from the eval to free time budget. The shipped score is the sliding window result.

## Architecture

| Component | Setting |
|-----------|---------|
| Physical layers | 12 |
| Effective layers | 16 (loop layers 4–5 × 3) |
| Model dim | 512 |
| Attention heads | 8 query, 4 KV (GQA) |
| MLP | 3.5× (hidden 1792), LeakyReLU(0.5)² |
| Tokenizer | sp9000 (custom 9000-vocab SentencePiece BPE) |
| Embeddings | Tied, SVD-scaled init (std=0.005), per-row absmax int8 |
| XSA | All layers |
| Parallel residuals | Layers 7+ |
| Skip connections | U-Net style with learned gates |
| Logit softcap | 30.0 |
| QK gain | 5.25 |
| RoPE base | 10000 |
| Partial RoPE | 16 / 64 dims (75% non-positional) |
| SHD | head-dim averaging on last 16 dims |
| LN scale | per-block scale parameter |
| AttnOutGate | enabled (zero-init) |
| SmearGate | enabled (zero-init) |

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer (matrices) | Muon (momentum=0.97, backend_steps=4, row-normalize) |
| Optimizer (scalars/embeddings) | AdamW (fused, β1=0.9, β2=0.95) |
| Matrix LR | 0.026 |
| Scalar LR | 0.025 |
| Tied embedding LR | 0.030 |
| Muon weight decay | 0.095 |
| Adam weight decay | 0.020 |
| Embed weight decay | 0.085 |
| EMA decay | 0.9965 |
| SWA | Last 33% of training, every 5 steps, 50/50 blend with EMA |
| Warmdown | Wallclock-based, frac=0.72 |
| Batch tokens | 786,432 |
| Sequence length | 2048 |
| Warmup steps | 20 |
| Loop schedule | NUM_LOOPS=2, LOOP_START=3, LOOP_END=5, ENABLE_LOOPING_AT=0.45 |
| Effective recurrence schedule | `[0,1,2,3,4,5,3,4,5,3,4,5,6,7,8,9,10,11]` (18 effective layers) |

## Quantization

GPTQ with SD-Clip scaling + asymmetric LQER post-quant correction:

| Parameter | Value |
|-----------|-------|
| Clip range | 31 (63 quantization levels) |
| Scale | k × std(row) / cr, k=15.0 |
| Embedding quant | int8 per-row absmax, k=20.0 |
| Embedding rank-1 residual | SVD rank-1 (u, v stored fp16 alongside int8 weights) |
| Calibration | 64 train-shard sequences (TRAIN_CALIB=1) |
| Compression | Brotli quality=11 + byte-shuffle stride=4 |
| Code compression | LZMA + base85 wrapper for submission.py |

The asymmetric LQER (low-rank quantization error recovery) packs A-matrix at INT2 + B-matrix at INT4 groupwise on the top-K worst-error tensors. Adds ~30 KB to the artifact, recovers ~0.001-0.003 BPB on local; pod gain expected larger due to ~6× larger pod quant gap.

Post-quant gap (pre-quant → post-quant): +0.0130 BPB. The gap is dominated by error compounding through the 16 effective recurrence layers reusing the same quantized weights.

Sliding window eval (stride=64) recovers −0.0164 BPB over post-quant via context-aware token-by-token rescoring.

## Reproduce

```bash
pip install -r requirements.txt
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Tokenizer + pre-tokenized data auto-downloads from [Idan3011/parameter-golf-sp9000](https://huggingface.co/datasets/Idan3011/parameter-golf-sp9000) on first run. All hyperparameters baked into `train_gpt.py` defaults — runs as-is with no env vars.

## What I Tried That Did Not Work (this iteration)

I tested each lever locally on RTX 4050 1800s smoke. The baseline pre-quant for the prior shipped stack on local was 1.3620; I required ≥ −0.005 pre-quant gain to ship.

| Lever | Source | Result | Note |
|---|---|---|---|
| Polar Express NS coefficients | PR #1344 / #1851 | DEAD: +0.013 pre-quant | Replaced fixed (3.4445, -4.775, 2.0315) with 5-tuple per-iter coefficients; regressed on this SHD+LERP architecture |
| Fused softcapped CE Triton kernel | PR #1851 | DEAD on 4050: +0.005 pre-quant, +2% step time | inductor likely already fuses softcap+CE; pod-only validation possible but I didn't pursue |
| MLP_GATE (per-token sigmoid on MLP output) | own | DEAD: +0.020 (thermal-confounded) | Slower steps + no signal |
| GATED_ATTN (per-layer full-dim output gate) | PR #1736 | wash | within thermal noise band |
| Recurrence depth curriculum (1→3→4) | PR #1756 | wash | |
| MUON_WD_MLP split (0.05 / 0.15) | own | wash / mild positive at 600s only | not confirmed at 1800s |
| EMB_BITS=7 | PR #1626 | inconclusive (thermal lottery) | 540 KB artifact savings real, pre-quant in noise |
| ROPE_DIMS=32 | own | thermal lottery confound | needs cleaner control run |
| MIN_LR=0.1 (LR floor) | PR #1787 | LOCAL WIN −0.005, NOT shipped this iter | reserved for next pod ship |
| LQER asymmetric rank-4 | PR #1530 / #1797 | shipped (in this artifact) | targets MLP fc weights with TOP_K=5, excludes tok_emb (already rank-1-corrected) |

## Previous Submission

| | Prior (1.1036) | This (1.0902) |
|---|---|---|
| Architecture | 12L 3.5× sp9000 | 12L 3.5× sp9000 |
| Effective layers | 16 | 16 |
| Step time | 139.8 ms | 132.0 ms |
| Steps | 4,292 | 4,544 |
| Pre-quant | 1.1033 | 1.0936 |
| Quant gap | +0.023 | +0.013 |
| Sliding gain over post-quant | −0.012 | −0.016 |
| Eval path | TTT (chunked + bigram hash) | sliding window |
| **Final val_bpb** | 1.1036 | **1.0902** |
| **Improvement** | — | **−0.0134** |

## Credits

- **LeakyReLU² activation**: PR #493 by @parinzee
- **AttnOutGate / SmearGate**: PR #1667, refined in PR #1693
- **LQER asymmetric rank-4 quantization correction**: PR #1530, asymmetric variant from PR #1797
