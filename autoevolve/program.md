# Parameter Golf Auto-Evolve: Agent Instructions

## Challenge Overview
Train the best language model that:
- Fits in a **16,000,000 byte** artifact (code + zlib-compressed INT8 model)
- Trains in under **10 minutes on 8xH100 SXM** GPUs
- Minimizes **val_bpb** (bits per byte) on FineWeb validation set
- Evaluation must also complete within 10 minutes (separate from training)
- No external downloads or network calls during eval - fully self-contained

## Leaderboard Context (as of 2026-03-19)
| Score  | Key Innovation                                     |
|--------|---------------------------------------------------|
| 1.1748 | FP16 embed + 10L + Muon WD + Overtone init + sliding eval |
| 1.1925 | Sliding window eval (stride=64) - pure eval technique |
| 1.1928 | LoRA test-time training (per-document)             |
| 1.2014 | 4K sequence length + NTK RoPE scaling              |
| 1.2060 | 2048 sequence length                               |
| 1.2147 | Mixed precision (int8/int6) quantization           |
| 1.2197 | FP16 tied embedding + LR/warmdown tuning           |
| 1.2244 | Naive baseline (9L 512dim)                         |

## Current SOTA Techniques (Already Implemented)
- **10 transformer layers** with U-Net skip connections (encoder/decoder halves)
- **512 model dim**, 8 heads, 4 KV heads (GQA), 2x MLP expansion (relu^2)
- **1024 vocab** (SentencePiece BPE), **1024 training seq len**
- **Tied embeddings in FP16** (not INT8) - halves quantization degradation
- **Overtone spectral init** - power-law SVD reshape of embedding matrix
- **Phase-transition residual mixing** - sigmoid schedule across layers
- **Muon optimizer** for matrix params + AdamW for embeddings/scalars
- **Decoupled weight decay** (0.02) on Muon matrix params
- **QK normalization** with learned per-head gain (init 1.5)
- **RoPE** with NTK-aware dynamic scaling for longer eval sequences
- **Logit softcap** at 30.0 (tanh-based)
- **CastedLinear** (fp32 weights, bf16 compute)
- **Sliding window eval** (stride=64, seq_len=1024)
- **INT8 per-row quantization + zlib** compression (level 9)
- ~15.4MB compressed artifact (~0.6MB headroom under 16MB)
- ~10,400 steps in 10 min (~57ms/step), ~5.5B tokens seen

## Ideas to Explore (Ranked by Expected Impact)

### Tier 1: High Impact (expected delta > -0.01)

1. **Depth Recurrence / Weight Tying**
   Share transformer weights across 2-3 forward passes through the same layers.
   Effectively doubles depth for zero extra parameters. Universal Transformers and
   recurrent transformer research show this works. Key: may need different residual
   scaling per pass, and step throughput drops proportionally.

2. **SwiGLU / GeGLU Activation**
   Replace relu^2 MLP with SwiGLU: out = (W1·x * silu(W3·x)) · W2.
   Consistently outperforms relu in LLaMA, Gemma, Mistral, etc.
   Trade-off: 3 matrices instead of 2 (gate proj), so reduce hidden dim to stay
   within parameter budget: hidden = round(2/3 * 4 * dim) for similar param count.

3. **Quantization-Aware Training (QAT)**
   Inject fake INT8 quantization noise during the last N% of training steps.
   The model learns to place weights in quantization-friendly ranges.
   Can reduce post-quant degradation from ~0.001 to near-zero BPB.

4. **Longer Training Context**
   Train at 2048+ tokens. More context = better predictions on longer dependencies.
   The 4K submission (1.2014) didn't combine with other SOTA improvements.
   Key: throughput drops ~2x at 2048, so fewer steps. Balance carefully.

5. **Improved Test-Time Training**
   LoRA TTT achieved 1.1928 but wasn't combined with SOTA training improvements.
   Combined with sliding window + current SOTA model = potentially large gains.
   Also: try rank-1 updates, per-document fine-tuning, chunk-level adaptation.

### Tier 2: Medium Impact (expected delta -0.003 to -0.01)

6. **Model Shape Optimization**: Try different dim/layers tradeoffs.
   E.g., 11 layers × 480 dim, or 9 layers × 544 dim. The optimal allocation of
   parameters between depth and width depends on the training budget.

7. **MLP Expansion Ratio**: Current 2x might be suboptimal. Try 2.5x or 3x with
   proportionally fewer layers or smaller dim.

8. **KV Head Count**: Currently 4. Try 2 (more parameter-efficient) or even 1 (MQA).
   Freed parameters can go to more layers or wider model.

9. **Learning Rate Sweep**: The LR landscape is complex (embed=0.6, tied=0.10,
   matrix=0.04, scalar=0.04). Small adjustments could unlock gains.

10. **Warmdown Schedule**: Currently 2500 iterations with wallclock-aware cosine decay.
    Try different warmdown fractions (0.3, 0.5, 0.7) or linear warmdown.

11. **Sliding Window Eval Stride**: Current stride=64. Try stride=32 or stride=16
    for better context coverage (at the cost of longer eval time).

### Tier 3: Experimental / Lower Confidence

12. **Mixture of Experts (MoE)**: Sparse activation for more capacity. Complex but
    could be very effective if the routing overhead is manageable.
13. **Layer-wise Learning Rates**: Different LRs for early vs late layers.
14. **Progressive Training**: Start with fewer layers, add more during training.
15. **Embedding Factorization**: Low-rank decomposition of the embedding matrix.
16. **Alternative Normalization**: DeepNorm, LayerScale, or QK-Norm variants.
17. **Better Initialization**: Xavier, Kaiming, or learned initialization patterns.
18. **Curriculum Learning**: Train on shorter/easier sequences first, increase over time.
19. **Gradient Checkpointing**: Trade memory for compute to enable larger model/batch.
20. **FP8 Training**: Mixed FP8/BF16 for faster matmuls on H100.

## Common Pitfalls
- **Artifact too large**: Adding parameters is easy; staying under 16MB is hard.
  The current model uses ~15.4MB. Only ~0.6MB headroom remains.
- **Training too slow**: Each extra ms/step costs ~17 training steps over 10 minutes.
  A 5ms increase costs ~850 steps. Balance capacity with throughput.
- **Post-quant degradation**: INT8 hurts small models. Keep embeddings in FP16.
- **Compile breaks**: torch.compile is fragile. Dynamic shapes, control flow, and
  some operations break compilation. Test carefully.
- **Multi-GPU bugs**: DDP/torchrun requires careful handling of per-rank operations.
- **Numerical instability**: Very aggressive hyperparameters can cause NaN/inf.

## How to Reason About Changes
1. **Parameter budget**: ~18.9M params → ~15.4MB compressed. ~0.6MB headroom.
2. **Step throughput**: ~57ms/step → ~10,400 steps in 10 min. Guard this carefully.
3. **One change at a time**: Isolate effects for clear signal on what works.
4. **Prioritize changes with strong theoretical backing** over random sweeps.
5. **Learn from history**: If hyperparameter X was tried and failed, try something else.
6. **Be creative**: The best gains come from novel architectural ideas, not tuning.
