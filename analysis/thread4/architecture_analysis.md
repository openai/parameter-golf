# Novel Architecture Analysis for Parameter Golf
## Thread 4: Architecture Investigation

**Date:** 2026-03-18  
**Objective:** Identify novel architectures that maximize val_bpb quality within 16MB artifact budget  
**Constraint:** ≤16MB artifact (int8+zlib), ≤10min on 8×H100, minimize FineWeb val_bpb  

---

## Executive Summary

After extensive literature review and quantitative parameter analysis, the **top 3 actionable recommendations** are:

| Rank | Architecture | Expected BPB Δ | Risk | Implementation Effort |
|------|-------------|----------------|------|----------------------|
| 1 | **SwiGLU drop-in** (9L d512 8:4, h=682) | -0.01 to -0.02 | Very Low | 5 lines |
| 2 | **9L d528 8:2 SwiGLU** (wider + fewer KV) | -0.02 to -0.03 | Low | 10 lines |
| 3 | **7x2 d672 SwiGLU** (proven recurrence + SwiGLU) | -0.03 to -0.05 | Medium | 15 lines |

---

## 1. Hybrid Attention-SSM (Mamba) Architecture

### Verdict: ❌ REJECT

### Analysis
Investigated Hymba (ICLR 2025), Jamba (ICLR 2025), and Nemotron-Flash (2025).

**Parameter cost problem:** At dim=512, a Mamba SSM layer requires **more parameters** than attention:
- Mamba (expand=2): 1,610,752 params/layer (2.05× attention)
- Mamba (expand=1): 805,376 params/layer (1.02× attention)
- GQA Attention (8:4): 786,432 params/layer

The SSM's `in_proj` (2×expand×dim²) dominates. GQA already reduces attention's KV cost significantly.

**Engineering blockers:**
1. Requires `causal_conv1d` and `mamba_ssm` custom CUDA extensions
2. Competition bans external downloads during eval → must include CUDA kernels in artifact
3. Adds compilation fragility and code size overhead
4. Mamba's advantage (linear-time sequence scaling) irrelevant at seq_len=1024

**Literature evidence:** Hymba showed SSM+attention hybrids outperform at 1.5B+ scale, but NO evidence of benefits at sub-20M params. SSM layers are designed for long sequences (4K+), not our 1024 context.

---

## 2. GQA Optimization (8:4 → 8:2 or 8:1)

### Verdict: ✅ STRONGLY RECOMMENDED

### Analysis
Current baseline uses 8 query heads with 4 KV heads (2:1 ratio).

| GQA Ratio | KV Params/Block | Total Savings | % of Model |
|-----------|----------------|---------------|------------|
| 8:4 (current) | 262,144 | 0 | 0% |
| 8:2 (recommended) | 131,072 | 1,179,648 | **6.9%** |
| 8:1 (MQA) | 65,536 | 1,769,472 | **10.4%** |

**Quality impact:**
- GQA paper (Ainslie et al. 2023): 4:1 ratio loses <0.5% quality vs MHA
- At head_dim=64, each KV head has enough capacity for meaningful attention patterns
- 8:2 (4:1 ratio) is safe; 8:1 (MQA) risks 1-2% quality loss

**What to do with freed 1.18M params:**
- **Option A:** Widen dim 512→528 (+16 dims across all layers) → better per-layer capacity
- **Option B:** Add 10th layer at dim≈496 → more depth
- **Option C:** Keep dim=512 with 1.2MB headroom for larger vocab or other innovations

**Recommendation:** Use 8:2 GQA and widen to dim=528 (net +3% more useful params).

---

## 3. SwiGLU vs ReLU² Activation

### Verdict: ✅ STRONGLY RECOMMENDED (FREE QUALITY IMPROVEMENT)

### Analysis
This is potentially the **single highest-impact, lowest-risk change** available.

**Current baseline:** ReLU²(x) = relu(xW_fc)² @ W_proj  
- 2 matrices: fc(dim→2×dim) + proj(2×dim→dim)  
- Params: 2 × dim × 2dim = 4dim² = 1,048,576/block

**SwiGLU:** SwiGLU(x) = (xW_gate ⊙ Swish(xW_up)) @ W_down  
- 3 matrices: gate(dim→h) + up(dim→h) + down(h→dim)  
- Params: 3 × dim × h

**Key insight:** Set h = 4dim/3 = 682 to MATCH ReLU² param count exactly:
- 3 × 512 × 682 = 1,047,552 ≈ 1,048,576 (ReLU²)

**Literature evidence for quality improvement:**
- Shazeer (2020): SwiGLU 4-5% better perplexity than ReLU/GELU at equal params
- Supernova (2025): Uses SwiGLU with 8/3 expansion for optimal parameter efficiency
- Llama, Mistral, Gemma, DeepSeek all use SwiGLU
- Conservative estimate for BPB: **-0.01 to -0.02 improvement**

**Implementation (5 lines):**
```python
class SwiGLUMLP(nn.Module):
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or int(4 * dim / 3)
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.up = CastedLinear(dim, hidden, bias=False)
        self.down = CastedLinear(hidden, dim, bias=False)
        self.down._zero_init = True
    
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

---

## 4. Untied Embeddings with Smaller LM Head

### Verdict: ❌ NOT RECOMMENDED at vocab=1024

### Analysis
The "Lost in Backpropagation" paper (arxiv 2603.10145, 2026) shows that tied embeddings create a gradient bottleneck where the softmax suppresses 95-99% of gradient norm through the LM head.

**However, at our scale (vocab=1024, dim=512), this is NOT a major issue:**
- LM head is 512→1024 (rank 512)
- Gradient compression ratio: 512/1024 = 50%
- Typical LLMs: 4096/50000 = 8% (much worse)
- Our bottleneck is **6.25× less severe** than typical LLMs

**Parameter cost of untying:**
- Full untied: +524,288 params (no room in budget)
- Bottleneck untied (b=128): +196,608 params
- Bottleneck untied (b=64): +98,304 params

**Conclusion:** The gradient bottleneck paper's findings apply primarily to large-vocab settings. At vocab=1024, the tied embedding is already nearly full-rank, and the param cost of untying isn't justified by the marginal gradient flow improvement.

---

## 5. Other Creative Architectures

### 5a. Supernova's Bottleneck Head Factorization
**Paper:** Supernova (arxiv 2507.15773, 2025)

Factors attention projections into shared bottleneck + per-head transforms:
W_Q^i = H_Q^i · B_Q where B is shared across heads.

Could save 40% of attention params, but:
- Complex to implement correctly
- Uncertain quality at dim=512 scale (designed for 650M+ params)
- The GQA already provides significant attention param savings
- **Status: Interesting but too speculative for now**

### 5b. PanGu-Pro Architectural Insights
**Paper:** PanGu-π Pro (ICML 2024)

Key findings for tiny language models:
- Wider MLP expansion ratios help small models more than depth
- Progressive training (small→large) can improve final quality
- **Actionable:** Supports our finding that wider > deeper at this scale

### 5c. Deep-Narrow with MQA
12 layers at dim=448 with 8:1 GQA:
- 15.5M params, fits in 14.5MB (1.5MB headroom!)
- Very deep (12 layers) but narrow
- **Risk:** dim=448 may be too narrow for good representation
- head_dim=56, which is below the typical 64 minimum
- **Status: Worth testing but high risk**

### 5d. Compression Advantage of Recurrence
**CRITICAL FINDING:** The 7x2 recurrence model (exp014) achieves 0.657 bytes/param compression vs baseline's 0.929 bytes/param. This is because:
1. Zero-initialized proj/down weights (41.5% of params) compress nearly to nothing with zlib
2. Weight sharing creates more uniform weight distributions
3. Per-iteration scalars are small and smooth

**This means recurrence models can store ~35% MORE params in the same 16MB budget!**
- Baseline: 17M params → 15.9MB
- Recurrence: 22.9M params → 15.1MB  

This compression advantage makes recurrence even more attractive.

---

## 6. Architecture Viability Matrix

All configs that fit in 16MB artifact:

| Architecture | Params | Artifact | Headroom | Risk | Expected BPB Δ |
|-------------|--------|----------|----------|------|----------------|
| BASELINE | 17.1M | 15.9MB | +0.1MB | REF | 0 |
| 9L d512 8:4 SwiGLU | 17.1M | 15.9MB | +0.1MB | Low | -0.01 to -0.02 |
| 9L d512 8:2 SwiGLU | 15.9M | 14.8MB | +1.2MB | Low | -0.01 to -0.02 |
| 9L d528 8:2 SwiGLU | 16.9M | 15.7MB | +0.3MB | Low | -0.02 to -0.03 |
| 10L d496 8:2 SwiGLU | 16.5M | 15.4MB | +0.6MB | Med | -0.02 to -0.03 |
| 10L d504 8:2 SwiGLU | 17.0M | 15.9MB | +0.1MB | Med | -0.02 to -0.04 |
| 11L d480 8:2 SwiGLU | 17.0M | 15.8MB | +0.2MB | Med | -0.01 to -0.03 |
| 7x2 d672 relu² (proven) | 22.9M | 15.1MB | +0.9MB | Med | **-0.018** (measured) |
| 7x2 d672 SwiGLU | 22.9M | 15.1MB | +0.9MB | Med | -0.03 to -0.05 |
| 7x2 d704 8:2 relu² | 23.5M | 15.5MB | +0.5MB | Med | -0.02 to -0.04 |

---

## 7. Recommended Experiment Priority

### Phase 1: Quick Ablations (2K steps, 1×A100)
1. **Exp-A:** SwiGLU drop-in on baseline (9L d512 8:4 SwiGLU h=682)
   - Expected: ~0.01 BPB better than baseline's 1.2963 at 2K
   - Implementation: Replace MLP class only
   
2. **Exp-B:** GQA reduction only (9L d512 8:2 relu²)
   - Expected: ~neutral, confirms no quality loss from fewer KV heads
   
3. **Exp-C:** Combined (9L d528 8:2 SwiGLU h=704)
   - Expected: ~0.02 BPB better, wider model via param reallocation

### Phase 2: Best Combo (2K steps)
4. **Exp-D:** 10L d504 8:2 SwiGLU (more depth)
5. **Exp-E:** 7x2 d672 8:4 SwiGLU (recurrence + SwiGLU)

### Phase 3: Full 8×H100 Run (10 min)
6. Best performing config from Phase 1-2

---

## 8. Key Research Citations

1. **Hymba** (ICLR 2025) - Hybrid attention+SSM heads. Viable at 1.5B+ but NOT at 17M scale.
2. **Supernova** (2025) - Bottleneck head factorization, SwiGLU, GQA optimization.
3. **PanGu-π Pro** (ICML 2024) - Architecture design for tiny LMs: wider > deeper.
4. **Lost in Backpropagation** (2026) - LM head gradient bottleneck. Not severe at vocab=1024.
5. **GQA** (Ainslie et al. 2023) - 4:1 ratio loses <0.5% quality.
6. **SwiGLU** (Shazeer 2020) - 4-5% perplexity improvement at matched params.
7. **Mamba** (Gu & Dao 2023) - SSM layers NOT parameter-efficient at small dim.

