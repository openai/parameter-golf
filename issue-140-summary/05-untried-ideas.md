# Untried Ideas & Combinations (Ranked by Expected Value)

## Tier 1 - High Expected Value

| Idea | Est. BPP Gain | Projected BPB | Notes |
|------|--------------|---------------|-------|
| **Reptile meta-TTT on #315's XSA+EMA base** | 0.003-0.008 | ~1.117-1.122 | Naive SGD TTT is dead on XSA+EMA (#303). Reptile modifies *training*, not just eval — may avoid disrupting EMA's weight landscape. #296 showed 0.011 BPB on SmearGate models. |
| **Sequence length curriculum** | 0.003-0.008 | — | Train at seq256-512 for first 20-30% (attention O(n^2) → 4x faster steps), ramp to full. DeepSpeed shows 2.2x token efficiency. Nobody has tried this. |

## Tier 2 - Top Picks

| Idea | Est. BPB Gain | Notes |
|------|--------------|-------|
| **Reptile meta-TTT** (PR #296) | 0.005-0.011 | 10x naive TTT on SmearGate. Zero artifact cost. Gain may partly be extra training steps. |
| **Mousse optimizer** (arXiv:2603.09697) | 0.003-0.008 | Curvature-aware Muon (Shampoo preconditioning). ~12% more effective at 3% overhead. Drop-in. |
| **Entropy-coded weights** (arXiv:2505.02380) | 0.003-0.008 | Replace zstd with ANS/Huffman. Could free 1-2 MB for more params. |
| **OptRot pre-quantization** (arXiv:2512.24124) | 0.002-0.005 | Rotation matrix redistributes outliers before quantizing. 30-50% int6 gap reduction. Zero artifact cost. Drop-in. |
| **Turbo-Muon** (arXiv:2512.04632) | 0.002-0.005 | Preconditioned Newton-Schulz. 5-10% faster training. |
| **HybridNorm** (arXiv:2503.04598) | 0.002-0.006 | Mixed Pre/Post-Norm for better depth utilization. Very low complexity. |
| **PPM-C mixing** (PR #283) | 0.003-0.008 | Classical compression blended with neural at eval. Low complexity. |
| **Differential Attention** (arXiv:2410.05258) | 0.005-0.015 | Difference of two softmax maps. High complexity (arch change). |
| **Batch size warmup** | 0.002-0.005 | Start 128-256K, ramp to 524K. Very low complexity. |
| **Cautious Weight Decay** (arXiv:2510.12402) | 0.001-0.004 | WD only where sign-aligned with gradient. Trivial (1 line). |
| **Gated Attention** (arXiv:2505.06708) | 0.002-0.005 | Per-head sigmoid gate eliminates attention sinks. ~2K params. |
| **Value Residual / ResFormer** (arXiv:2410.17897) | 0.002-0.006 | Layer-1 value shortcut. 20% fewer tokens needed. ~11 scalars. |
| **WaveletGPT** (arXiv:2409.12924) | 0.003-0.010 | Multi-scale Haar wavelet. 40-60% faster convergence. Zero params. |

## Tier 3 - Novel, Higher Risk

| Idea | Est. BPB | Notes |
|------|----------|-------|
| **Knowledge distillation** | ~1.125-1.145 | Train larger teacher ~7 min, distill into 16MB student ~3 min. High complexity. |
| **Mixture of Experts** (PR #250) | ~1.120-1.140 | 4 SwiGLU experts with hybrid routing. Awaiting compute. Wide uncertainty. |
| **Partial weight sharing + 14L** | net +0.005-0.015 | Share middle-layer pairs with LoRA adapters. Saves 3-5 MB → 14 layers. |
| **nGPT hypersphere normalization** (arXiv:2410.01131) | 0.003-0.008 | Q/K on unit-norm hypersphere. Eliminates extreme Q condition numbers. 4-20x convergence claim. |
| **BitNet + sliding window + SmearGate** (PR #139) | ~1.16 | 64.5M ternary params at 1.2029 + eval tricks. Fundamental quality gap vs int6. |
