# Experiment Log: 12L + Low-Rank Q + QAT + FTLE + Stride-OGD

**Hardware: 1xH100 80GB HBM3 (development/testing)**
**Target: 8xH100 SXM for final submission**

## Plan

Combine 4 novel techniques that nobody has combined in the competition yet:
1. **Low-Rank Q + 12 layers**: Low-Rank Q factorization (rank=128) gives ~8% faster steps per layer, funding 12 layers instead of 10
2. **QAT with STE**: Quantization-Aware Training with straight-through estimator reduces quant gap from ~0.016 to ~0.005 BPB
3. **FTLE-guided per-row precision**: Instead of blanket int6 for middle layers, use accumulated gradient sensitivity (FTLE) to allocate precision per-row. Hot rows get int6-7, cold rows get int4-5
4. **Stride-OGD at eval**: Online gradient descent on a 1024-dim vocab bias during stride-64 sliding window eval — free BPB improvement

## Size Budget Analysis

- 12L with Low-Rank Q (r=128): ~19.4M params
- Need mixed precision to fit in 16MB
- Target: avg ~int5.5 effective bits → ~15MB compressed
- fp16 embedding (tied) stays at 1.0MB

---

## Log

### 2026-03-21 03:20 UTC — Project kickoff (1xH100)
- Analyzed current SOTA: 1.1748 bpb (10L, sliding window, fp16 embed, Muon WD, overtone init)
- Analyzed int6 mixed precision record: 1.2147 bpb (10L, int8/int6 mixed)
- Designed combined approach targeting 12L + all 4 techniques
- Created record directory, beginning implementation
- Data download started (10 shards for dev testing)

### 2026-03-21 03:25 UTC — Implementation start (1xH100)
- Writing train_gpt.py based on current SOTA script
- Adding Low-Rank Q, QAT, FTLE gradient tracking, per-row precision quantization, Stride-OGD eval
