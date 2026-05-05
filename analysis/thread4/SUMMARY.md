# Thread 4 Summary: Novel Architecture Investigation

## Files Created
- `architecture_analysis.md` — Full 8-section analysis report
- `swiglu_implementation.py` — Drop-in SwiGLU MLP replacement code
- `SUMMARY.md` — This file

## Key Findings

### 1. SwiGLU Activation ✅ HIGHEST PRIORITY
- **Drop-in replacement for ReLU²** with <1% param increase
- SwiGLU(h=688) at dim=512: 1,056,768 params vs ReLU² 1,048,576 (+0.8%)
- Literature: 2-5% perplexity improvement at matched params
- Used by ALL modern LLMs (Llama, Mistral, Gemma, etc.)
- **Expected BPB improvement: -0.01 to -0.02**
- Implementation: 5 lines, replace MLP class

### 2. GQA 8:4 → 8:2 ✅ RECOMMENDED
- Saves 1,179,648 params (6.9% of model)
- Quality loss <0.5% at 4:1 ratio
- Freed params enable: dim 512→528 or +1 layer at dim≈496
- **Expected net BPB: -0.01 to -0.02 from better param allocation**

### 3. Mamba/SSM Hybrid ❌ REJECTED
- MORE params than attention at dim=512 (2.05× with expand=2)
- Requires custom CUDA extensions (competition-incompatible)
- SSM advantage only at seq_len >> 1024

### 4. Untied Embeddings ❌ NOT WORTH IT
- Gradient bottleneck not severe at vocab=1024 (50% vs typical 8%)
- Would cost 98K-524K extra params

### 5. Recurrence Compression Advantage 🔑 KEY INSIGHT
- Recurrent models compress to 0.657 bytes/param vs baseline 0.929
- Can fit ~35% MORE params in same 16MB budget
- Zero-init weights (41.5% of model) compress ~100:1 with zlib

## Recommended Experiments (Priority Order)

| # | Config | Param Change | Expected BPB Δ | Risk |
|---|--------|-------------|----------------|------|
| 1 | SwiGLU drop-in (9L d512 8:4 h=688) | +0.8% | -0.01 to -0.02 | Very Low |
| 2 | 9L d528 8:2 SwiGLU (h=704) | -1.1% | -0.02 to -0.03 | Low |
| 3 | 10L d504 8:2 SwiGLU (h=672) | -0.1% | -0.02 to -0.04 | Medium |
| 4 | 7x2 d672 8:4 SwiGLU (h=896) | +0.8%* | -0.03 to -0.05 | Medium |
| 5 | 11L d480 8:2 SwiGLU (h=640) | -0.4% | -0.01 to -0.03 | Medium |

*Recurrence models fit despite more params due to better compression.

## Best Combined Architecture Estimate
**9L d528 8:2 SwiGLU** (non-recurrence) or **7x2 d672 SwiGLU** (recurrence)
could achieve **val_bpb ≈ 1.20-1.21** (beating baseline 1.2244 by 0.01-0.02).
