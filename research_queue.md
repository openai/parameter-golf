# Research Queue
# Updated: 2026-04-06 (initial seed)
# Discards since last keep: 0
# Current best BPB: 1.2490

## Next Up

### Full Hessian GPTQ with AR Self-Gen Calibration
- Reasoning: Our naive per-row int6 quantization loses significant precision. Full Hessian GPTQ with Cholesky compensation is the single biggest technique gap vs SOTA (1.1147).
- Changes: Replace `quantize_int6_per_row` (line 545) with full GPTQ implementation. Add AR generation for calibration data post-training.
- Radicality: significant
- Mechanism: Better quantization → less information loss during int6 compression → lower post-quant BPB
- Category: quantization
- Est. BPB gain: 0.01-0.03

### SWA (Stochastic Weight Averaging)
- Reasoning: SOTA uses "EMA(0.997) + Tight SWA(every 50)". We have EMA but not SWA. SWA averages weights at regular intervals during warmdown, complementing EMA.
- Changes: Add SWA collection every 50 steps during warmdown phase. Average collected checkpoints before export.
- Radicality: moderate
- Mechanism: SWA → wider optima → better generalization → lower val_bpb
- Category: optimizer
- Est. BPB gain: 0.002-0.005

### Selective Pruning Before Quantization
- Reasoning: SOTA prunes ±1 values by reconstruction error. Removing high-error weights before quantization reduces overall quantization noise.
- Changes: After training, before quantization: identify weights where int6 rounding causes high error, set to ±1 instead.
- Radicality: moderate
- Mechanism: Fewer high-error quantized weights → lower reconstruction loss → better post-quant BPB
- Category: quantization
- Est. BPB gain: 0.003-0.008

## Queue

### BigramHash 3072×112
- Current: 2048×128. SOTA: 3072×112. Wider but thinner.
- Category: architecture

### Flash Attention 3
- Hopper warp-specialized kernels → more steps/second → more training
- Category: architecture

### Warmdown Scaling
- Scale warmdown_iters proportionally to step count
- Category: schedule

## Rejected
(none yet)
