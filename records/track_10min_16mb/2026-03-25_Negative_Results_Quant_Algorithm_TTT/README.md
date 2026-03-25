# Non-Record: Negative Results — Quantization Algorithms & TTT on Val-GPTQ Stack

6 experiments on the current SOTA stack (1.1142 BPB, val-calibrated GPTQ + XSA-all + BigramHash 3072), all negative or neutral.

**Base:** Val-Calibrated GPTQ + XSA-all + BigramHash 3072 — 1.1142 BPB (3-seed mean), 8×H100 SXM, 600s

---

## Quantization Algorithm Experiments (All Negative)

Tested whether the GPTQ algorithm itself could be improved. Both methods use the same trained weights and int6 grid — they only change how rounding decisions are made.

| Approach | Sliding BPB | vs Baseline | Why It Failed |
|----------|-------------|-------------|---------------|
| Baseline GPTQ (control) | 1.1139 | — | Standard column-wise GPTQ with val-calibrated Hessians |
| Qronos iterative Hessian (3 iters) | 1.1146 | +0.0007 (worse) | Re-collects H = X^T X after each layer is quantized, so later layers see quantized activations. At int6 the per-layer error is so small (~0.0003 BPB) that iterating doesn't help — the updated Hessians are nearly identical to the original ones. |
| CDQuant coordinate descent (3 passes) | 1.1144 | +0.0005 (worse) | After GPTQ, revisits each weight and tries flipping its rounding direction. At int6 with 63 levels, the grid spacing is ~0.06 scale units — most weights are already at their optimal grid point. The coordinate descent finds almost nothing to flip. |

**Conclusion:** At int6, the quantization gap is only +0.0036 BPB. Column-wise GPTQ is already near-optimal at this bit-width. Iterative Hessian correction (Qronos) and post-hoc rounding refinement (CDQuant) are designed for aggressive 2-4 bit quantization where cross-layer error compounds. At 6-bit, the error per layer is too small for these methods to improve on.

---

## Test-Time Training Experiments (All Negative)

Legal score-first TTT: score each chunk under `inference_mode()`, THEN train on it. Every token is graded before any adaptation. This is the same protocol that was legal in [PR #549](https://github.com/openai/parameter-golf/pull/549).

| Approach | Params Unfrozen | TTT BPB | Non-TTT Baseline | Delta | Eval Time |
|----------|-----------------|---------|-------------------|-------|-----------|
| Full TTT (all params) | 27.1M (100%) | 1.1146 | 1.1145 | +0.0001 (worse) | 445s |
| MLP-down-only | 8.7M (32%) | 1.1145 | 1.1144 | +0.0001 (neutral) | 424s |
| MLP-all (up + down) | 17.3M (64%) | 1.1144 | 1.1143 | +0.0001 (neutral) | 422s |

TTT hyperparameters: lr=0.002, epochs=3, chunk_tokens=32768, stride=64.

**Conclusion:** TTT does not help on the val-calibrated GPTQ + XSA-all + BigramHash 3072 stack. This is now **25 total failed TTT attempts** across two stacks:
- 22 on PR #593 stack (1.1171 BPB) — documented in [PR #670](https://github.com/openai/parameter-golf/pull/670)
- 3 on val-GPTQ stack (1.1142 BPB) — this work

**Why TTT keeps failing:**
1. **Score-first constraint:** The model must score each chunk before adapting. Early tokens in each chunk get zero benefit from adaptation.
2. **Val-calibrated GPTQ interaction:** GPTQ rounding decisions are optimized for val activation patterns. TTT gradient updates shift the dequantized weights away from those optimized rounding points, potentially undoing the val-calibration advantage.
3. **Catastrophic forgetting at chunk boundaries:** Each 32K-token chunk resets to the base model. The 3-epoch training per chunk overfits to local patterns that don't generalize to the scoring window.
4. **The base model is already good:** At 1.1142 BPB, the model's predictions are strong enough that test-time adaptation introduces more noise than signal.

---

## Meta-Lessons

1. **GPTQ algorithm is near-optimal at int6.** The remaining quant headroom is in WHAT you quantize to (the grid), not HOW you assign values to grid points (the algorithm).

2. **TTT is dead on this stack.** 25 experiments, zero positive results. The val-calibrated GPTQ + XSA-all combination leaves no room for eval-time weight adaptation.

3. **Seed variance (~0.0003-0.0007 BPB) dominates.** Most "improvements" from quant algorithm tweaks or TTT are within noise. Only techniques that move BPB by >0.001 are distinguishable from random variation.
