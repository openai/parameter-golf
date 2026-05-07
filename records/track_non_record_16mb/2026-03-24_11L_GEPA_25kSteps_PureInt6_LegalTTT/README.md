# 11L GEPA + 25k Steps + Pure Int6 + 15-Percentile GPTQ-lite + Legal TTT

**val_bpb = 1.0944** | Pre-TTT float: 1.1088 | TTT gain: **−0.014** | Artifact: 13.75 MB | Submission: 13.83 MB

> Non-record unlimited-compute submission (trained on 4×A100-40GB, eval 2074s on 4×A100).

---

## Headline Result

This submission sets a new personal best of **1.0944 BPP** by extending training to **25000 steps** (12000 peak-LR + 13000 warmdown) with pure int6 quantization and legal score-first TTT. The 13000-step warmdown drives the float base to **1.1088** — the lowest we have achieved — while simultaneously producing the **smallest artifact** at 13.75 MB.

The result of **1.0944 BPP** surpasses the current record-track #1 (signalrush PR #414: 1.1228) by **0.028 BPP** (noting this is a non-record unlimited-compute submission) and improves on our own 20k result (1.0983) by **0.004 BPP**.

---

## Novel & Creative Contributions

### 1. 25000-Step Training with 13000-Step Warmdown

Training for 25000 steps with cosine warmdown starting at step 12000. The key finding from our scaling series is that the peak-LR plateau saturates around 1.216–1.222 BPP (by ~step 9000), but warmdown length produces nearly linear returns in float-base improvement. The 13000-step warmdown drives the float base from ~1.218 down to **1.1088** — a gain of **0.109 BPP** during warmdown alone.

The warmdown acceleration is dramatic in the final 5000 steps:

| Step | val_bpb | Phase |
|---|---|---|
| 20000 | 1.1689 | Mid-warmdown |
| 21000 | 1.1615 | −7.4/kstep |
| 22000 | 1.1507 | −10.8/kstep |
| 23000 | 1.1387 | −12.0/kstep |
| 24000 | 1.1243 | −14.4/kstep |
| 24500 | 1.1156 | −17.4/kstep |
| 25000 | 1.1088 | **−13.6/kstep** |

The warmdown *accelerates* as it approaches zero LR — the improvement rate in the final 3000 steps (−14.0/kstep) exceeds the mid-warmdown rate (−7.4/kstep at step 21000). This acceleration occurs even as the cosine schedule's LR changes decelerate toward zero, suggesting fine-grained optimization at low LR is disproportionately effective.

### 2. Confirmed Scaling Law: Every Metric Improves with Steps

| Steps | Float Base | TTT BPP | Artifact Size | ΔBPB/kstep |
|---|---|---|---|---|
| 9000 | 1.1353 | 1.1157 | 14.94 MB | baseline |
| 12000 | 1.1268 | 1.1079 | 14.79 MB | −2.6/kstep |
| 15000 | 1.1217 | 1.1035 | 14.52 MB | −1.5/kstep |
| 20000 | 1.1153 | 1.0983 | 14.22 MB | −1.0/kstep |
| **25000** | **1.1088** | **1.0944** | **13.75 MB** | **−0.8/kstep** |

Three trends hold perfectly across all 5 experiments:
1. **Float base improves** (1.135 → 1.109)
2. **TTT BPP improves** (1.116 → 1.094)
3. **Artifact shrinks** (14.94 → 13.75 MB)

Diminishing returns are significant (~0.8 BPP/kstep at 25k vs ~2.6 at 12k) but still clearly positive.

### 3. TTT Gain Compression at Lower Base BPP

The TTT gain narrows as the float base improves:

| Float Base | TTT BPP | TTT Gain |
|---|---|---|
| 1.135 | 1.116 | −0.019 |
| 1.127 | 1.108 | −0.019 |
| 1.122 | 1.104 | −0.018 |
| 1.115 | 1.098 | −0.017 |
| **1.1088** | **1.0944** | **−0.014** |

The TTT gain has compressed from −0.019 to −0.014, suggesting better-trained models leave less room for TTT improvement. At this rate, the TTT gain may approach −0.010 for very long training runs.

### 4. Pure Int6 Per-Row Quantization with 15-Candidate GPTQ-lite

All model tensors (including embeddings) use int6 per-row quantization with GPTQ-lite optimal clip search across 15 percentile candidates. Combined with zstd-22 compression, the 27M-parameter model compresses to just **13.75 MB** — the smallest artifact across all experiments. The trend of smaller artifacts with longer training is remarkably consistent.

### 5. Legal Score-First TTT with SGD Momentum

Competition-legal test-time training using sliding windows with score-first protocol. SGD with momentum 0.9, lr=0.002, 10 epochs per 32K-token chunk, first 2 blocks frozen, 5% LR warmup.

---

## Architecture Summary

| Component | Configuration |
|---|---|
| Layers | 11 |
| Embedding dim | 512 |
| Heads | 8 query, 4 KV (GQA) |
| MLP | 3× expansion (1536 hidden), ReLU² activation |
| Vocab | 1024 (SentencePiece BPE) |
| BigramHash | 2048 buckets, 128-dim embeddings |
| RoPE | Partial: 16/64 dims, YARN scaling (train_seq=2048) |
| Value Embeddings | 128d on layers 9–10, per-layer scale (init 0.1) |
| LN Scale | `1/√(layer+1)` depth scaling |
| XSA | Cross-sequence attention on last 4 layers |
| U-Net skips | Residual connections across layer pairs |
| SmearGate | Learned token-mixing gate on input embeddings |
| Tied Embeddings | Yes |
| Parameters | 27,030,107 total |

## Training Details

| Setting | Value |
|---|---|
| Hardware | 4×A100-40GB (NVIDIA) |
| Steps | 25,000 |
| Peak-LR phase | Steps 0–12,000 |
| Warmdown | Cosine anneal from step 12,000 to 25,000 (13,000 steps) |
| Warmup | 20 steps |
| Batch size | 786,432 tokens |
| Sequence length | 2,048 |
| Matrix LR (Muon) | 0.025 |
| Scalar LR (Adam) | 0.025 |
| Embed LR | 0.035 |
| Decoder LR mult | 2.0× |
| Weight decay | 0.04 (both Muon and Adam) |
| Grad clip | 0.3 |
| EMA decay | 0.997 |
| Late QAT | Disabled (hurts compression) |
| Training wallclock | 12,509 seconds (~3h 28m) |
| Step average | 500.4 ms |

## Quantization Details

| Setting | Value |
|---|---|
| Method | Int6 per-row with GPTQ-lite clip search |
| GPTQ-lite candidates | 15 percentiles per row |
| Compression | zstd level 22 |
| Embedding quant | Int6 (QUANT_EMBED=1) |
| Mixed quant | Disabled (MIXED_QUANT=0) |
| Model bytes | 14,418,655 |
| Code bytes | 78,281 |
| **Total submission** | **14,496,936 bytes (13.83 MB)** |

## TTT (Test-Time Training) Details

| Setting | Value |
|---|---|
| Protocol | Score-first (legal) |
| Optimizer | SGD |
| Learning rate | 0.002 with cosine decay + 5% warmup |
| Momentum | 0.9 |
| Epochs per chunk | 10 |
| Chunk size | 32,768 tokens |
| Stride | 64 tokens |
| Frozen blocks | First 2 of 11 |
| Gradient clip | 1.0 |
| Total chunks | 1,893 |
| Eval time | 2,074 seconds |

## Training Trajectory

| Step | val_bpb | Phase |
|---|---|---|
| 0 | 4.1043 | Init |
| 500 | 1.3944 | Warmup complete |
| 1000 | 1.3250 | Peak LR |
| 2000 | 1.2655 | Peak LR |
| 3000 | 1.2456 | Peak LR |
| 5000 | 1.2305 | Peak LR |
| 7000 | 1.2256 | Peak LR |
| 9000 | 1.2200 | Peak LR |
| 10000 | 1.2176 | Peak LR |
| 11000 | 1.2163 | Peak LR (plateau) |
| 12000 | 1.2175 | Warmdown start |
| 13000 | 1.2096 | Warmdown |
| 14000 | 1.2054 | Warmdown |
| 15000 | 1.1990 | Warmdown |
| 16000 | 1.1951 | Warmdown |
| 17000 | 1.1890 | Warmdown |
| 18000 | 1.1826 | Warmdown |
| 19000 | 1.1771 | Warmdown |
| 20000 | 1.1689 | Warmdown |
| 21000 | 1.1615 | Warmdown |
| 22000 | 1.1507 | Warmdown accelerating |
| 23000 | 1.1387 | Warmdown accelerating |
| 24000 | 1.1243 | Warmdown accelerating |
| 24500 | 1.1156 | Almost done |
| 25000 | **1.1088** | **Final float** |

## Comparison with All Results

| Run | Steps | Float | TTT BPP | Artifact | Total | Status |
|---|---|---|---|---|---|---|
| **This (exp_25k)** | **25000** | **1.109** | **1.0944** | **13.75 MB** | **13.83 MB** | **✅ NEW BEST** |
| exp_20k_pur | 20000 | 1.115 | 1.0983 | 14.22 MB | 14.29 MB | ✅ Previous best |
| exp_15k_pur | 15000 | 1.122 | 1.1035 | 14.52 MB | 14.60 MB | ✅ |
| exp_12k_pur | 12000 | 1.127 | 1.1079 | 14.79 MB | 14.87 MB | ✅ |
| exp_9k_pure | 9000 | 1.135 | 1.1157 | 14.94 MB | 15.02 MB | ✅ |
| signalrush (SOTA) | 7051 | 1.142 | 1.1228 | 15.55 MB | N/A | Record track |

---

## Acknowledgments

This submission builds on techniques introduced by many contributors to the parameter-golf community:

- **signalrush** (PR #414): GPTQ-lite clip search and EMA — the quantization backbone of this submission
- **jfprincz** (PR #315): Partial RoPE (16/64 dims) and layerwise LN scale
- **jfprincz** (PR #287): XSA on last 4 layers, EMA replacing SWA, MLP 3× expansion
- **unnir** (PR #265): Efficient Partial XSA concept
- **raahilshah** (PR #162): SmearGate, BigramHash embeddings, OrthoInit, Muon weight decay
- **aruniyer** (PR #86): Int6 quantization with STE QAT
- **samacqua**: LoRA-based test-time training concept
- **abaybektursun** (PR #549): LeakyReLU² activation exploration
- **OpenAI**: Baseline architecture, Muon optimizer, and competition infrastructure

---

## What's Next: 30k and 40k Experiments

Based on the consistent scaling law, we have 30k and 40k experiments queued:

| Experiment | Steps | Warmdown | Projected Float | Projected TTT |
|---|---|---|---|---|
| 30k | 30,000 | 18,000 | ~1.100–1.105 | ~1.085–1.090 |
| 40k | 40,000 | 22,000 | ~1.095–1.100 | ~1.080–1.085 |

The diminishing returns curve suggests we may see ~0.004–0.006 BPP gain from 25k → 30k and another ~0.003–0.005 from 30k → 40k. Beyond 40k, the returns will likely become marginal.
