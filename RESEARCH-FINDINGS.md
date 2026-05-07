# Parameter Golf Research Findings
## Novel Techniques for Extremely Small but Powerful Language Models

**Researcher:** Prometheus (DeepSeek V3.2)
**Date:** March 23, 2026
**Competition Context:** OpenAI Parameter Golf - Train best LM under 16MB total artifact + 10 min on 8×H100 GPUs. Scored by bits-per-byte (bpb). Current #1: 1.1228 bpb using 11 transformer layers, int6 quantization, GPTQ-lite, EMA, XSA, Partial RoPE, BigramHash, SmearGate.

---

## 1. BitNet / 1-bit LLMs

### What is it?
BitNet is Microsoft's work on binary/ternary weight networks that use 1.58-bit weights (ternary: -1, 0, +1). The key insight is that neural networks can operate with extremely low-bit weights while maintaining performance through careful training techniques.

### Application to Parameter Golf
- Replace some or all transformer layers with 1.58-bit ternary layers
- Could reduce weight storage by ~16x compared to float16 (1.58 bits vs 16 bits)
- Particularly effective for feed-forward networks where weight matrices are large
- Combine with existing int6 quantization for hybrid precision

### Expected Impact on bpb
- **High potential:** Could reduce model size by 3-5x while maintaining similar perplexity
- Estimated bpb improvement: 0.05-0.15 reduction (from ~1.12 to ~0.97-1.07)
- Risk: Training stability issues with extreme quantization

### Implementation Complexity
- **Medium-Hard:** Requires modified training pipeline with gradient scaling
- Need to implement ternary weight constraints during forward/backward passes
- Existing implementations available but need adaptation for transformer architecture

### Sources
- Microsoft BitNet papers (2023-2024)
- "1.58-bit LLMs" research from Microsoft Research
- GitHub: `microsoft/BitNet` (implementation references)

---

## 2. MatMul-free Language Models

### What is it?
A revolutionary architecture that completely eliminates matrix multiplication operations from LLMs. Uses ternary weights (+1, 0, -1) and replaces MatMul with simple addition and negation operations. The "Scalable MatMul-free Language Modeling" paper (June 2024) shows 2.7B parameter models without MatMul achieving similar performance to conventional LLMs.

### Application to Parameter Golf
- Eliminate the most computationally expensive operation in transformers
- Replace dense layers with ternary linear layers using addition/negation
- Particularly valuable for attention mechanisms and feed-forward networks
- Could enable much larger models within the 16MB constraint

### Expected Impact on bpb
- **Very High potential:** Could enable 2-3x more parameters within same size budget
- Estimated bpb improvement: 0.08-0.20 reduction
- Major win if it works at small scale

### Implementation Complexity
- **Hard:** Requires complete reimplementation of transformer blocks
- Need to design MatMul-free attention and feed-forward layers
- Reference implementation: `github.com/ridgerchu/matmulfreellm`

### Sources
- arXiv:2406.02528 "Scalable MatMul-free Language Modeling" (June 2024)
- GitHub: `ridgerchu/matmulfreellm`
- EmergentMind topic: "Matrix Multiplication-Free LM"

---

## 3. Product Quantization for Neural Networks

### What is it?
Codebook-based weight compression where weights are represented as combinations of learned codewords from a shared codebook. Each weight vector is approximated as a sum of a few codewords, dramatically reducing storage.

### Application to Parameter Golf
- Apply product quantization to weight matrices in transformer layers
- Use small codebooks (e.g., 256 entries) shared across layers
- Particularly effective for large weight matrices in feed-forward networks
- Can be combined with other quantization techniques

### Expected Impact on bpb
- **Medium-High potential:** 4-8x compression for weight matrices
- Estimated bpb improvement: 0.03-0.10 reduction
- Works well with already-quantized weights

### Implementation Complexity
- **Medium:** Need to implement codebook learning and quantization during training
- Inference requires codebook lookups and summation
- Reference: "Quantization Meets Projection" (arXiv:2411.06158, Nov 2024)

### Sources
- arXiv:2411.06158 "Quantization Meets Projection: A Happy Marriage for Codebook-Based Quantization"
- "EPQuant: A Graph Neural Network compression approach based on product quantization"
- GitHub: `Efficient-ML/Awesome-Model-Quantization`

---

## 4. Depth Recurrence / Universal Transformers

### What is it?
Running the same transformer layers multiple times (recurrent in depth) instead of stacking unique layers. Universal Transformers share parameters across depth, allowing deeper processing without additional parameters.

### Application to Parameter Golf
- Use 2-4 shared transformer blocks repeated 3-6 times
- Dynamic halting mechanism to adapt computation per token
- Effectively get 12-24 layers of computation with only 2-4 layers of parameters
- Perfect for extreme parameter constraints

### Expected Impact on bpb
- **High potential:** 3-6x more computation with same parameters
- Estimated bpb improvement: 0.06-0.15 reduction
- Risk: Training convergence issues with deep recurrence

### Implementation Complexity
- **Medium:** Modify transformer to support recurrent depth
- Implement adaptive computation time (ACT) for dynamic halting
- Reference: Universal Transformers paper (ICLR 2019, updated 2025)

### Sources
- "Universal Transformers" (arXiv:1807.03819, updated 2025)
- "Depth-Recurrent Transformer" (EmergentMind topic)
- "Efficient Parallel Samplers for Recurrent-Depth Models" (arXiv:2510.14961, Oct 2025)

---

## 5. Mixture of Experts at Tiny Scale

### What is it?
Sparse Mixture of Experts (MoE) where different experts handle different inputs, but only a few experts are active per token. Traditionally used for large models, but recent work explores tiny-scale MoE.

### Application to Parameter Golf
- Create 4-8 tiny experts (e.g., 1M parameters each) with router selecting 2
- Share most parameters across experts (e.g., attention layers)
- Use extremely sparse routing (top-1 or top-2)
- Could provide specialization without parameter bloat

### Expected Impact on bpb
- **Medium potential:** Better modeling capacity with modest parameter increase
- Estimated bpb improvement: 0.02-0.08 reduction
- Risk: Routing overhead and training instability at small scale

### Implementation Complexity
- **Medium-Hard:** Implement sparse gating and expert parallelism
- Need careful initialization and load balancing
- Reference: "Mixture of Experts in Large Language Models" survey (arXiv:2507.11181, Dec 2025)

### Sources
- arXiv:2507.11181 "Mixture of Experts in Large Language Models" (Dec 2025)
- arXiv:2407.06204 "A Survey on Mixture of Experts in Large Language Models"
- "Dynamic Mixture of Experts: An Auto-Tuning..." (ICLR 2025)

---

## 6. Hyper-networks

### What is it?
Small networks that generate weights for a larger main network. A hypernetwork takes some input (e.g., layer index, token embedding) and outputs the weights for that layer.

### Application to Parameter Golf
- Use tiny hypernetwork (100K-500K parameters) to generate all transformer weights
- Input: layer index + positional encoding → output: layer weights
- Drastically reduces stored parameters (only hypernetwork weights stored)
- Can generate different weights per layer, per head, etc.

### Expected Impact on bpb
- **Very High potential:** Could reduce parameters by 10-100x
- Estimated bpb improvement: 0.10-0.30 reduction
- Risk: Expressive power limitations and training difficulty

### Implementation Complexity
- **Hard:** Design hypernetwork architecture and training procedure
- Need to ensure generated weights have sufficient expressivity
- Reference: "Hypernetworks: Meta-Modeling in Deep Learning" (EmergentMind)

### Sources
- "Hypernetworks — A novel way to initialize weights" (Medium, Jan 2025)
- arXiv:2306.06955 "A Brief Review of Hypernetworks in Deep Learning"
- "An open dataset of neural networks for hypernetwork research" (arXiv:2507.15869, Jul 2025)

---

## 7. KAN (Kolmogorov-Arnold Networks)

### What is it?
Alternative to MLP layers based on Kolmogorov-Arnold representation theorem. KANs have learnable activation functions on edges instead of fixed activations on nodes. More parameter-efficient for function approximation.

### Application to Parameter Golf
- Replace MLP blocks in transformers with KAN layers
- Potentially more expressive per parameter than standard MLPs
- Could use smaller hidden dimensions for same modeling power
- Particularly useful in feed-forward networks

### Expected Impact on bpb
- **Medium potential:** Better modeling capacity with same parameters
- Estimated bpb improvement: 0.03-0.08 reduction
- Risk: Slower training and inference, less optimized

### Implementation Complexity
- **Medium:** Implement KAN layers with spline-based activation functions
- Need efficient forward/backward passes for spline functions
- Reference: "KAN: Kolmogorov-Arnold Networks" (arXiv:2404.19756, ICLR 2025)

### Sources
- arXiv:2404.19756 "KAN: Kolmogorov-Arnold Networks" (ICLR 2025)
- OpenReview: "KAN: Kolmogorov–Arnold Networks"
- "Kolmogorov–Arnold graph neural networks for molecular property prediction" (Nature Machine Intelligence, Aug 2025)

---

## 8. Neural Network Weight Compression (2024-2026)

### What is it?
Latest extreme compression techniques including CRVQ (Channel-Relaxed Vector Quantization), additive quantization, and sub-2-bit methods that enable near-lossless 1-bit compression.

### Application to Parameter Golf
- CRVQ: 38.9% improvement over strongest sub-2-bit PTQ baseline
- Additive quantization for extreme compression (2-3 bits per parameter)
- Model phase transitions analysis to find compression limits
- Combine multiple compression techniques

### Expected Impact on bpb
- **High potential:** State-of-the-art compression beyond int6/GPTQ
- Estimated bpb improvement: 0.05-0.12 reduction
- Direct improvement over current #1's quantization approach

### Implementation Complexity
- **Medium:** Implement latest quantization algorithms
- Need calibration data and compression-aware training
- Reference: "CRVQ: Channel-Relaxed Vector Quantization for Extreme Compression of LLMs" (TACL 2025)

### Sources
- "CRVQ: Channel-Relaxed Vector Quantization for Extreme Compression of LLMs" (TACL, Nov 2025)
- "Extreme compression of large language models via additive quantization" (ICML 2024)
- "Bielik-Q2-Sharp: A Comparative Study of Extreme 2-bit Quantization" (arXiv:2603.04162, Mar 2026)
- GitHub: `whucs21Mzy/Model-Phase-Transitions`

---

## 9. Test-time Training / Test-time Compute

### What is it?
Adapting the model at inference time based on the input sequence. Uses a small amount of computation per sequence to specialize the model, improving performance without changing stored parameters.

### Application to Parameter Golf
- Use first 100-1000 tokens of each sequence to adapt model
- Lightweight adaptation (e.g., bias tuning, prompt tuning)
- Could significantly improve perplexity on test data
- Particularly valuable for domain adaptation

### Expected Impact on bpb
- **Medium potential:** Better adaptation to test distribution
- Estimated bpb improvement: 0.04-0.10 reduction
- Risk: Computation overhead counts against 10-minute limit

### Implementation Complexity
- **Medium:** Implement online adaptation during inference
- Need careful design to avoid overfitting to adaptation tokens
- Reference: "You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs" (arXiv:2510.10223, Oct 2025)

### Sources
- arXiv:2510.10223 "You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs" (Oct 2025)
- GitHub: `Dereck0602/Awesome_Test_Time_LLMs`
- "Test-Time Training on Nearest Neighbors for Large Language Models" (ICLR 2024)

---

## 10. Novel Small-Model Architectures

### What is it?
Non-transformer architectures that beat transformers at small scale, including RWKV (RNN+Transformer hybrid), Mamba (State Space Models), and other efficient alternatives.

### Application to Parameter Golf
- **RWKV-7 "Goose":** RNN with transformer-like performance, linear time, constant space
- **Mamba-3:** Improved SSM with better discretization and efficient inference
- **xLSTM 7B:** Efficient mLSTM-based model
- These architectures are inherently more parameter-efficient

### Expected Impact on bpb
- **Very High potential:** Architectural advantage over transformers at small scale
- Estimated bpb improvement: 0.10-0.25 reduction
- Could be the biggest breakthrough if it works

### Implementation Complexity
- **Hard:** Implement completely different architecture
- Need to adapt training pipeline for RWKV/Mamba
- Reference implementations available but need optimization

### Sources
- GitHub: `BlinkDL/RWKV-LM` (RWKV-7 "Goose", Mar 2025)
- "RWKV: Reinventing RNNs for the Transformer Era" (arXiv:2305.13048)
- GitHub: `state-spaces/mamba` (Mamba SSM)
- "Mamba-3: Improved Sequence Modeling using State Space Principles" (OpenReview, Oct 2025)
- "xLSTM 7B" (Beck et al., 2025)

---

## Synthesis & Recommendations

### Highest Potential Techniques (Tier 1)
1. **MatMul-free Language Models** - Biggest architectural win if it works
2. **RWKV/Mamba architectures** - Inherent efficiency advantages
3. **Hyper-networks** - Extreme parameter reduction potential
4. **BitNet/1.58-bit weights** - Direct improvement over current quantization

### Medium Potential Techniques (Tier 2)
5. **Depth Recurrence** - More computation with same parameters
6. **Product Quantization** - Better compression than int6
7. **Extreme Compression (CRVQ)** - State-of-the-art quantization
8. **Test-time Adaptation** - Free improvement at inference

### Lower Risk Techniques (Tier 3)
9. **KAN layers** - Moderate improvement to MLPs
10. **Tiny-scale MoE** - Specialization without major overhead

### Implementation Strategy
1. **Start with RWKV/Mamba** - Test if architectural advantage materializes at <16MB
2. **Add MatMul-free layers** - If RWKV shows promise, integrate MatMul-free concepts
3. **Apply extreme quantization** - Use CRVQ or BitNet-style quantization
4. **Incorporate depth recurrence** - Share parameters across layers
5. **Add test-time adaptation** - Final tuning for competition data

### Expected Combined Impact
If multiple techniques work synergistically:
- **Conservative estimate:** 1.1228 → 0.95-1.00 bpb (10-15% improvement)
- **Aggressive estimate:** 1.1228 → 0.80-0.90 bpb (20-30% improvement)
- **Breakthrough potential:** <0.80 bpb with novel architecture + extreme compression

### Next Steps
1. Implement and test RWKV-7 at small scale (<16MB)
2. Experiment with MatMul-free transformer blocks
3. Benchmark extreme quantization techniques (CRVQ, BitNet)
4. Explore hyper-network weight generation
5. Combine best approaches into integrated architecture

---

## References & Resources

### Key Papers
1. **MatMul-free:** arXiv:2406.02528 (Jun 2024)
2. **BitNet:** Microsoft Research papers (2023-2024)
3. **RWKV-7:** GitHub `BlinkDL/RWKV-LM` (Mar 2025)
4. **Mamba-3:** OpenReview `HwCvaJOiCj` (Oct 2025)
5. **CRVQ:** TACL 2025 "Channel-Relaxed Vector Quantization"
6. **Hypernetworks:** arXiv:2306.06955 (2024 review)
7. **KAN:** arXiv:2404.19756 (ICLR 2025)
8. **Universal Transformers:** arXiv:1807.03819 (updated 2025)
9. **Product Quantization:** arXiv:2411.06158 (Nov 2024)
10. **Test-time Adaptation:** arXiv:2510.10223 (Oct 2025)

### Code Repositories
- `github.com/ridgerchu/matmulfreellm` - MatMul-free implementation
- `github.com/BlinkDL/RWKV-LM` - RWKV architecture
- `github.com/state-spaces/mamba` - Mamba SSM
- `github.com/Efficient-ML/Awesome-Model-Quantization` - Quantization resources
- `github.com/Dereck0602/Awesome_Test_Time_LLMs` - Test-time adaptation

### Research Trends (2025-2026)
- **Small models outperforming large:** Domain-specific fine-tuning beats general models
- **Architectural diversity:** RWKV, Mamba, xLSTM challenging transformer dominance
- **Extreme compression:** Sub-2-bit quantization becoming practical
- **Efficient training:** MatMul-free and other compute-reduction techniques
- **Adaptive inference:** Test-time computation for better performance

---

**Conclusion:** The Parameter Golf competition is ripe for disruption. Current leaders use optimized transformer variants, but novel architectures (RWKV/Mamba), MatMul-free computation, and hyper-networks offer order-of-magnitude efficiency improvements. Combining architectural innovation with extreme compression could achieve bpb well below 1.0, potentially winning the