# BitNet b1.58 Ternary v2 — 68M Params, val_bpb=1.1770

**Non-record submission.** Improves on PR #139 (1.2029 -> 1.1770) and documents systematic findings on what works and what breaks for ternary models in this competition. All BPB numbers on the leaderboard use sliding window eval (stride=64); we report the same for comparability.

## Results

| Metric | Value |
|---|---|
| Standard val_bpb | 1.1983 |
| Ternary roundtrip val_bpb | 1.1999 |
| **Sliding window val_bpb (stride=64)** | **1.1770** |
| Roundtrip gap | 0.0016 |
| Artifact size | 15,882,274 bytes (15.88 MB) |
| Training steps | 4,591 in 600s (130.7 ms/step) |
| Model params | 68M ternary |

Despite 2.4x more parameters than the best int6 submissions, our ternary model underperforms the int6 SOTA by ~0.05 BPB on sliding window eval (1.1770 vs 1.127). The gap is primarily due to convergence speed — ternary STE noise slows training, and most techniques that help int6 models are incompatible with ternary (see below). We believe ternary becomes more competitive at longer training budgets where the parameter count advantage dominates.

## Negative Results Summary

The most useful finding from this work. Negative results below are from ablation runs during v1/v2 development. The included train.log is from the final v2 run only.

| Technique | int6 effect | Ternary effect | Root cause |
|---|---|---|---|
| XSA | +0.002 | plateau at 2.4 | Ternary attention too coarse for self-bias |
| Weight decay | +0.003 | plateau at 2.4 | Fights STE, causes sparsity collapse |
| Grad clipping | standard | stalls at step 500 | Double-normalizes with optimizer |
| SmearGate | +0.01 | -0.02 | Can't adapt to modified inputs |
| OrthoInit | enables SmearGate | no effect | Destroyed by quantization |
| EMA/SWA | +0.005 | broken (0.14 gap) | Averaging ternary = non-ternary |
| TTT | +0.01-0.03 | no effect | Not enough signal to shift 68M coarse params |

## Architecture

- 12 transformer layers, 768 dim, 12 heads, 6 KV heads (GQA)
- MLP 3.25x (hidden 2496), relu-squared activation
- BitLinear for all attention + MLP projections (ternary {-1, 0, 1} with per-group absmax STE)
- U-Net skip connections, tied fp16 embeddings
- RoPE base 200,000, logit softcap 30.0

## What Worked

**Wider MLP (2304 -> 2496).** Ternary packing at 1.6 bits/param makes extra parameters almost free in the artifact. Widening the MLP from 3x to 3.25x added 4.4M params but only 0.8MB to the artifact. Direct improvement: 1.2029 -> 1.1983 (standard eval).

**Higher learning rate (0.04 vs competition consensus of 0.02-0.025).** We hypothesize that ternary STE gradients are inherently noisier than int6 — the quantization function snaps continuous weights to just 3 levels — and higher LR helps the optimizer overcome this noise floor. The competition consensus LR was tuned for int6 models where quantization noise is much smaller.

**fp16 scale simulation during training.** Training computes scales in fp32, but serialization stores them in fp16. Different precision means different rounding means different ternary values, which caused a 0.05 BPB gap in early experiments. Training with `.half().float()` on scales simulates fp16 precision during training, closing the gap to 0.0016 BPB. This is the single most important technique for ternary roundtrip fidelity.

**Longer warmdown.** The final ~600 steps of training (LR decaying linearly to 0) were the most productive phase, improving BPB by 0.022 (step 4000: 1.2201 -> step 4591: 1.1983). We suspect that at low LR, the continuous shadow weights converge to values that round cleanly to {-1, 0, 1}, reducing quantization error.

**Base-3 packing (1.6 bits/param).** Ternary values {-1, 0, 1} are encoded as base-3: 5 trits per byte. This achieves 1.6 bits/param, near the information-theoretic minimum of log2(3) = 1.585 bits. Combined with LZMA compression, 68M ternary params + fp16 scales fit in 15.82MB (+ 63KB code = 15.88MB total).

## What Broke the Network

These techniques caused the model to plateau at val_loss around 2.4 (vs normal convergence to around 2.0). Results from ablation runs during development.

**XSA (Exclusive Self-Attention).** XSA subtracts the self-value projection from attention output, removing the "self-attention bias" where tokens attend too much to themselves. For int6 models this is a free +0.002 BPB. For ternary, it causes a complete training plateau. We believe ternary attention weights are too coarse to exhibit self-attention bias — the attention patterns are already sparse and noisy, so removing the self-value component removes signal, not noise.

**Weight decay (0.04).** Decoupled weight decay shrinks continuous weights toward zero. For int6, this improves quantization robustness and generalization. For ternary, it causes training collapse. We suspect that shrinking weights toward zero causes more weights to quantize to 0 instead of +/-1, making the model progressively sparser until it loses capacity. The ternary STE may already provide sufficient implicit regularization.

**Gradient clipping (0.3-1.0).** The optimizer already normalizes gradients via Newton-Schulz orthogonalization. Adding gradient clipping on top double-normalizes the signal. With 68M params, the global gradient norm is naturally large. Clipping at 1.0 caused training to stall after around 500 steps.

**EMA/SWA.** Fundamentally incompatible with ternary. Averaging ternary-quantized weights produces non-ternary values, destroying the quantization structure. Confirmed in v1 experiments with a 0.14 BPB gap.

## What Didn't Help

**SmearGate + BigramHash.** SmearGate blends each token's embedding with the previous token's. BigramHash adds a hash-table embedding for token pairs. Together they give +0.01 BPB for int6 models. For ternary, they hurt by ~0.02 BPB regardless of initialization. We believe the ternary weights can't adapt to the modified input distribution — the quantization is too coarse to learn the subtle adjustments needed.

**OrthoInit.** Orthogonal weight initialization is required for SmearGate to work in int6 models. For ternary, the carefully constructed orthogonal structure is destroyed on the first forward pass when weights snap to {-1, 0, 1}. Default Kaiming initialization works equally well.

**Test-time training (TTT).** Causal SGD adaptation during eval (evaluate chunk, record scores, train on chunk, repeat). Gives +0.01-0.03 BPB for int6 models. For ternary, no improvement in any variant tested — frozen ternary (only updating norms/scales), all params, higher LR than competition standard (2e-3 vs 3e-4). The 62M-token val set doesn't provide enough signal to meaningfully shift a 68M-param ternary model without destabilizing it. The coarse weight structure means the loss landscape around the dequantized weights isn't smooth enough for few-step SGD adaptation.

## QAT Insight

Our full-training STE (quantize every forward from step 0) achieves near-zero roundtrip gap (0.0016 BPB) but hurts convergence — the model trains slower because every gradient is corrupted by quantization noise. The competition's "Late QAT" approach (enable STE only in the last 15% of training) is likely optimal: full-precision convergence for 85% of training, then STE to close the quantization gap.

This suggests an unexplored direction: int4 with late QAT. Int4 gives 50% more params than int6 (32M vs 21M) in the same budget. With late QAT, the quantization gap should be near-zero — our ternary full-STE gap is only 0.0016, and int4 with 16 levels would be even smaller. We have not tested this.

## Run Command

```bash
bash run_8xh100.sh
```

## References

- BitNet b1.58: [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)
- Prior submission: PR #139 (val_bpb=1.2029)
