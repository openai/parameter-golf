# Parameter Golf — Approach Notes

## Strategy Overview

Maximize language model quality within a 16MB artifact constraint and 10 minutes on 8×H100s. Five pillars informed by research in model compression, efficient architectures, and training optimization.

---

## 1. Depth Recurrence (Layer Sharing)

Instead of unique parameters per layer, reuse a small set of transformer blocks recursively. A 4-block recursive model with 8 passes achieves the effective depth of a 32-layer network while only storing 4 layers of parameters.

Research shows recursive transformers achieve comparable loss to standard architectures with 3-4× fewer parameters. The model learns to refine representations through repeated application of the same weights — a form of iterative refinement that naturally suits the extreme parameter constraint.

**Target:** Replace 12 unique layers with 4 recursive blocks × 3 passes = 12 effective layers at 1/3 the parameter cost.

## 2. Factorized Embeddings

The embedding matrix is often the largest single component. Instead of a full V×H matrix, decompose it into V×E and E×H where E << H. This technique (from ALBERT) can reduce embedding parameters by 80%+ while maintaining representation quality.

Combined with tied input/output embeddings, this eliminates the output projection layer entirely — the same factorized embedding serves both input and output.

**Math:** At vocab 1024, hidden 512: Full = 524K params. Factorized (E=128): 131K + 65K = 196K params. Savings: 63%.

## 3. Quantization-Aware Training (QAT)

Train the model knowing it will be quantized. The model learns weight distributions that survive low-precision conversion. At 2-bit precision, 16MB supports ~32M parameters.

Key insight: post-training quantization at 2-bit loses 15-20% quality. QAT at 2-bit loses only ~4%. The difference is massive at this scale.

**Approach:** Train at FP16/BF16, apply QAT during training with straight-through estimators, export at 2-bit for the final artifact.

## 4. Knowledge Distillation

Use a larger pretrained model as a teacher during training. The 8×H100 budget can run a 7B teacher alongside a 32M student. The student learns from soft probability distributions rather than hard labels, capturing more knowledge per training step.

Distillation is especially powerful for small models — the teacher provides a richer gradient signal than raw cross-entropy on token predictions alone.

## 5. Training Maximization

Every second of the 10-minute budget matters:

- **Sequence packing:** Multiple short examples per input sequence, no wasted padding tokens
- **Curriculum ordering:** Train on FineWeb examples ordered by difficulty (shorter/simpler first, longer/complex later) for faster initial convergence
- **Cosine LR schedule:** High initial learning rate with cosine decay over the 10-minute window
- **Gradient accumulation:** Effective batch size tuned for optimal loss curves on H100s
- **Mixed precision training:** BF16 compute for speed, QAT checkpoints for artifact size

## 6. Tokenizer Optimization

Vocabulary size directly impacts embedding parameter count. The baseline uses 1024 tokens. Exploring:

- Smaller BPE vocabularies (512, 256) — fewer embedding parameters but worse compression
- The tradeoff is parameter cost vs bytes-per-token — the evaluation metric is bits per byte, so better compression from larger vocab can offset the parameter cost
- Custom tokenizer trained specifically on FineWeb distribution

## 7. Alternative Architectures

Beyond standard transformers:

- **State-space models (Mamba-style):** Linear scaling with sequence length, potentially more parameter-efficient for the same quality
- **Mixture of Experts at micro-scale:** Multiple tiny FFN experts with a router — only a subset active per token, more capacity per parameter
- **Depth-adaptive inference:** Early exit for easy tokens, full depth for hard ones — maximizes quality where it matters most

---

## The Math

| Bitwidth | Parameters in 16MB | Architecture |
|----------|-------------------|-------------|
| 2-bit | ~32M | Recursive transformer, factorized embeddings |
| 3-bit | ~21M | Standard transformer, tied embeddings |
| 4-bit | ~16M | Compact transformer |

## Experiment Plan

- [ ] Run baseline (9-layer, 512-dim, 1024-vocab, tied embeddings) — establish score to beat (1.2244)
- [ ] Implement depth recurrence (4 recursive blocks × 3 passes)
- [ ] Add factorized embeddings (V×128 + 128×H)
- [ ] Test 2-bit QAT during training
- [ ] Knowledge distillation with 7B teacher
- [ ] Curriculum data ordering on FineWeb
- [ ] Tokenizer vocabulary sweep (256, 512, 1024, 2048)
- [ ] Mamba/SSM architecture comparison
- [ ] Combine best techniques into final submission

## Background

5 production fine-tuned models (7B-72B) deployed via QLoRA/GGUF/NVFP4 quantization on NVIDIA DGX hardware. Built a 130K-chunk expert knowledge base for AI/ML research consultation. Deep experience with compression-quality tradeoffs across bitwidths.

## Status

Credits requested. Local experimentation with MLX baseline in progress.
