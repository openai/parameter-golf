# SP8192 + Depth Recurrence + Muon 0.99 + SmearGate + EMA

## Score

| Metric | Value |
|--------|-------|
| **Pre-Quantization BPB** | **1.1497** |
| **Post-Quantization BPB** | **1.3936** |
| Training Steps | 10,301 |
| Training Time | 599,988 ms (10 min cap) |
| Artifact Size | 16,077,239 bytes (< 16 MiB) |
| Peak Memory | 41,806 MiB / 78,710 MiB |

## Techniques

1. **SP8192 Vocabulary Scaling** — Switched from the default 1024-token BPE to an 8192-token SentencePiece vocabulary for superior text compression per token.

2. **Depth Recurrence** — Re-run transformer layers 4 and 5 a second time during the forward pass, creating an 11-layer virtual architecture from 9 physical layers with zero additional parameters.

3. **Muon Optimizer Tuning** — Increased momentum from the default 0.95 to **0.99** with a **1500-step warmup** schedule (starting at 0.85) for aggressive convergence under the 10-minute time constraint.

4. **Extended Warmdown** — Increased warmdown from 1200 to **3000 steps** for smoother final weight distributions.

5. **SmearGate** — A learned per-dimension sigmoid gate after the embedding layer that blends each token's representation with its predecessor, capturing bigram statistics at the embedding level.

6. **EMA (Exponential Moving Average)** — Shadow weight tracker (decay=0.996) that smooths training volatility. EMA weights are swapped in before final serialization.

7. **Sliding Window Evaluation** — Replaced naive non-overlapping chunk evaluation with stride-64 sliding windows, granting 960+ tokens of context per scored position.

8. **Decoupled Weight Decay** — Separate weight decay for Muon matrices (0.085), embedding parameters (0.085), and Adam scalars (0.02).

9. **Artifact Compression** — INT8 per-row quantization with zlib compression to fit under the 16MB limit.

## Known Issue: Quantization Penalty

The pre-quantization BPB of **1.1497** demonstrates competitive architecture performance. However, depth recurrence causes W² gradient amplification over 10,000 steps, creating large magnitude outliers in the weight matrices. Standard INT8 per-row clipping allocates excessive dynamic range to these outliers, effectively zeroing out the majority of smaller weights.

**Planned fix:** MuonEq-R (row-normalized Muon) to prevent outlier growth during training, combined with Full-Hessian GPTQ for Cholesky error compensation during quantization.

## Platform

RunPod 8×H100 80GB SXM
