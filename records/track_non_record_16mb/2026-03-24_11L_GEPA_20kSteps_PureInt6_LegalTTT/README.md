# 11L GEPA + 20k Steps + Pure Int6 + 15-Percentile GPTQ-lite + Legal TTT

**val_bpb = 1.0983** | Pre-TTT float: 1.1153 | Int6 quant: ~1.142 | TTT gain: **−0.044** | Artifact: 14.22 MB | Submission: 14.29 MB

> Non-record unlimited-compute submission (trained on 4×A100-40GB, eval 2066s on 4×A100).

---

## Headline Result

This submission pushes BPP **below 1.10 with legal score-first TTT** by extending training to **20000 steps** (12000 peak-LR + 8000 warmdown) with pure int6 quantization. The 8000-step warmdown produces a spectacular float base decline from 1.216 → 1.115, demonstrating that **warmdown length is the dominant lever** for improving the float base model.

The result of **1.0983 BPP** surpasses the current record-track #1 (signalrush PR #414: 1.1228) by **0.0245 BPP** (noting this is a non-record unlimited-compute submission) and improves on our own 15k result (1.1035) by **0.005 BPP**.

---

## Novel & Creative Contributions

### 1. 20000-Step Training with 8000-Step Warmdown

Training for 20000 steps with a cosine warmdown starting at step 12000. The key insight from our scaling experiments is that the peak-LR plateau saturates around 1.216-1.222 BPP (by ~step 9000), but the warmdown phase produces a steep, nearly linear decline in loss. The 8000-step warmdown drives the float base from ~1.216 down to **1.1153** — a gain of **0.101 BPP** during warmdown alone.

Confirmed scaling law from our experiments:

| Steps | Peak-LR | Warmdown | Float Base | TTT BPP | Gain/kstep |
|---|---|---|---|---|---|
| 9000 | 5000 | 4000 | 1.135 | 1.116 | baseline |
| 12000 | 7000 | 5000 | 1.127 | 1.108 | −2.7/kstep |
| 15000 | 9000 | 6000 | 1.122 | 1.104 | −1.3/kstep |
| **20000** | **12000** | **8000** | **1.115** | **1.098** | **−1.2/kstep** |

Diminishing returns are significant but the model keeps improving. The warmdown phase accounts for the majority of the gains.

### 2. Pure Int6 Per-Row Quantization with 15-Candidate GPTQ-lite

All model tensors (including embeddings) use int6 per-row quantization with GPTQ-lite optimal clip search across 15 percentile candidates. Combined with zstd-22 compression, the 27M-parameter model compresses to just **14.22 MB** of model data — the smallest artifact across all our experiments (longer training → smaller artifact due to better-conditioned weights).

### 3. Legal Score-First TTT with SGD Momentum + LR Warmup

Competition-legal test-time training using sliding windows with score-first protocol. SGD with momentum 0.9, lr=0.002, 10 epochs per 32K-token chunk, first 2 blocks frozen, 5% LR warmup.

TTT gain of **−0.044 BPP** (slightly less than the −0.046 at 12k steps), consistent with the pattern that better-trained base models leave less room for TTT improvement.

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
| Steps | 20,000 |
| Warmdown | Cosine anneal from step 12,000 to 20,000 |
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

## Quantization Details

| Setting | Value |
|---|---|
| Method | Int6 per-row with GPTQ-lite clip search |
| GPTQ-lite candidates | 15 percentiles per row |
| Compression | zstd level 22 |
| Embedding quant | Int6 (QUANT_EMBED=1) |
| Mixed quant | Disabled (MIXED_QUANT=0) |
| Model bytes | 14,907,461 |
| Code bytes | 78,281 |
| **Total submission** | **14,985,742 bytes (14.29 MB)** |

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
| Eval time | 2,066 seconds |

## Training Trajectory

| Step | val_bpb | Phase |
|---|---|---|
| 0 | 4.104 | Init |
| 500 | 1.394 | Warmup complete |
| 1000 | 1.323 | Peak LR |
| 2000 | 1.265 | Peak LR |
| 3000 | 1.247 | Peak LR |
| 5000 | 1.231 | Peak LR |
| 7000 | 1.226 | Peak LR |
| 9000 | 1.219 | Peak LR |
| 10000 | 1.217 | Peak LR |
| 11000 | 1.217 | Peak LR |
| 12000 | 1.216 | Warmdown start |
| 13000 | 1.206 | Warmdown |
| 14000 | 1.199 | Warmdown |
| 15000 | 1.189 | Warmdown |
| 16000 | 1.181 | Warmdown |
| 17000 | 1.169 | Warmdown |
| 18000 | 1.154 | Warmdown |
| 19000 | 1.136 | Warmdown |
| 19500 | 1.125 | Warmdown |
| 20000 | **1.115** | **Final float** |

## Comparison with All Results

| Run | Steps | Float | Quant | TTT BPP | Artifact | Status |
|---|---|---|---|---|---|---|
| **This (exp_20k_pur)** | **20000** | **1.115** | **~1.142** | **1.0983** | **14.29 MB** | **✅ NEW BEST** |
| exp_15k_pur | 15000 | 1.122 | ~1.149 | 1.1035 | 14.52 MB | ✅ |
| exp_12k_pur | 12000 | 1.127 | ~1.154 | 1.1079 | 14.79 MB | ✅ |
| exp_9k_pure | 9000 | 1.135 | ~1.162 | 1.1157 | 14.94 MB | ✅ |
| gep_9k3k | 9000 | 1.136 | ~1.162 | 1.1153 | 15.97 MB | ✅ |
| signalrush (SOTA) | 7051 | 1.142 | 1.149 | 1.1228 | 15.55 MB | Record track |

---

## Research Arc: What We Learned from the Non-Record Track

This submission is the culmination of a week-long series of non-record experiments (March 18–24, 2026) that explored what really matters for parameter golf once the 10-minute training constraint is removed. Below we distill five transferable findings and frame the open questions they point toward.

### Finding 1: Warmdown Is a First-Class Research Variable, Not Cleanup

The training trajectory table above makes this vivid: the model plateaus near 1.216–1.219 BPB across steps 9000–12000 at peak LR, then drops **0.101 BPB** during the 8000-step warmdown to reach 1.115. The late peak-LR phase (steps 7000–12000) delivers only ~0.010 BPB over 5000 steps — a rate of ~2 BPB/kstep. Warmdown delivers 12.6 BPB/kstep, roughly **6× the late-plateau rate**.

| Phase | Steps | ΔBPB | BPB/kstep |
|-------|-------|------|-----------|
| Early peak-LR (1k→7k) | 6,000 | −0.097 | −16.2/kstep |
| Late peak-LR plateau (7k→12k) | 5,000 | −0.010 | −2.0/kstep |
| Warmdown (12k→20k) | 8,000 | −0.101 | **−12.6/kstep** |

The early peak-LR phase is faster per step (as expected), but the model hits a wall around step 7000. Warmdown breaks through that wall. This isn't "cleanup" — once the plateau sets in, warmdown is where a large fraction of remaining gain originates. For record-track submissions limited to ~7k total steps, optimizing the warmdown-to-peak-LR ratio deserves at least as much attention as architecture changes.

### Finding 2: Better-Trained Models Are Easier to Compress

A counterintuitive but consistent result: the longest-trained model produces the **smallest artifact**.

| Steps | Float BPB | Artifact Size | Float→Final ΔBPB |
|-------|-----------|---------------|-------------------|
| 9,000 | 1.135 | 14.94 MB | −0.019 |
| 12,000 | 1.127 | 14.79 MB | −0.019 |
| 15,000 | 1.122 | 14.52 MB | −0.019 |
| **20,000** | **1.115** | **14.29 MB** | **−0.017** |

(Note: Float→Final ΔBPB measures the gap between the unquantized float model and the final TTT output, encompassing both quantization damage and TTT recovery.)

This suggests that optimization quality improves weight compressibility, not just floating-point loss. The warmdown phase appears to organize weight distributions into lower-entropy configurations that compress better under int6 + zstd. This is directly relevant to the 16 MB artifact constraint: the "best" model might also be the smallest.

### Finding 3: Legal Full-Model TTT Prefers SGD Over AdamW in This Regime

The AdamW → SGD transition across the March 22–23 runs produced our cleanest transferable conclusion. Both configurations below use the same 5.2k-step, 24.6M-parameter base model (float BPB ~1.161), so the TTT gains are directly comparable:

| TTT Config | Optimizer | Epochs | Frozen | Float→Final ΔBPB | Source |
|------------|-----------|--------|--------|-------------------|--------|
| Full-model | AdamW, lr=5e-5 | 1 | 0 layers | −0.007 | PR #456 (Mar 22) |
| Freeze-2 | SGD, lr=0.002, mom=0.9 | 3 | 2 layers | −0.017 | PR #461 (Mar 22) |
| Freeze-2 | SGD, lr=0.002, mom=0.9 | 30 | 2 layers | −0.018 | Mar 23 (30ep) |

SGD delivers **2.4× the TTT gain** of AdamW on the identical base model. The mechanism is straightforward: AdamW's second-moment estimates cannot converge when each chunk provides only ~30 gradient steps, while SGD + momentum's simpler dynamics are better matched to this short-horizon fitting problem.

Separately, on the 20k GEPA model (27M params, different architecture), SGD TTT with 10 epochs recovers −0.044 BPB from the quantized baseline — but this larger number reflects both a different model family and a different measurement baseline (quant→final rather than float→final), so it should not be directly compared to the AdamW numbers above.

### Finding 4: Freezing Early Layers During TTT Is Useful Regularization, Not Just Safety

Freezing the first 2 of 11 blocks (~18% of depth) during TTT isn't merely defense against catastrophic forgetting — it's actively beneficial. Early layers encode generic lexical and syntactic features that are shared across all FineWeb domains. Later layers are the better adaptation surface because they hold more task/domain-specific representations.

This gives practitioners a concrete, mechanistic TTT heuristic: **freeze early layers proportional to their generality, adapt later layers proportional to their specificity.** The AdamW→SGD+freeze comparison (Finding 3) confirms this: even though freezing removes parameters from TTT, the resulting model adapts better.

### Finding 5: After the Right TTT Family, Invest in the Base Model

The marginal value of TTT tuning shrinks once the base recipe is strong. Looking at float→final gain as a share of total improvement over the naive baseline (1.224):

| Base Float BPB | Final BPP | Float→Final ΔBPB | Total Gain vs Baseline | TTT Share |
|----------------|-----------|-------------------|------------------------|-----------|
| 1.161 (5.2k steps) | 1.143 | −0.018 | 0.081 | 22% |
| 1.135 (9k GEPA) | 1.116 | −0.019 | 0.108 | 18% |
| 1.127 (12k GEPA) | 1.108 | −0.019 | 0.116 | 16% |
| 1.115 (20k GEPA) | 1.098 | −0.017 | 0.126 | 13% |

(Total Gain = naive baseline 1.224 − final BPP. TTT Share = Float→Final ΔBPB / Total Gain.)

As the base model improves, TTT's percentage contribution shrinks from 22% to 13%. The really big jump in our recipe family came from choosing the right TTT regime (SGD + freeze + multi-epoch), not from endlessly polishing it. After that, additional base model quality delivers more BPB per unit of effort than exotic TTT micro-tuning.

---

## What Transfers to the Record Track

Based on our non-record experiments, we believe the following are directly transferable to the 10-minute constraint:

**Likely transferable:**
- **Warmdown emphasis** — allocate a larger fraction of total steps to warmdown (our best results use ≥40%)
- **GPTQ-lite / pure int6** — the quantization pipeline works regardless of training duration
- **SGD-based legal TTT** — the 2.4× gain over AdamW holds on the same base and should transfer
- **Freeze-early-blocks** — a simple, robust TTT regularization heuristic

**Less transferable:**
- Very long training curves (20k steps requires ~2.8 hours on 4×A100)
- Large eval-time TTT budgets (10–30 epochs/chunk → 2000–3600s eval)
- Gains that only appear because eval can take ~2000s instead of 600s

---

## Open Frontiers

The local TTT recipe (SGD, freeze count, epoch count, per-layer LR) appears mostly saturated for the current protocol. The next meaningful questions are structural:

1. **Stream vs. document-based adaptation** — should TTT state reset per document/topic?
2. **Self-distillation at test time** — can teacher signals improve adaptation?
3. **Quantization-aware TTT** — can adaptation be made aware of int6 rounding?
4. **Base-training scaling laws under fixed 16 MB budget** — formalizing the warmdown/compression/TTT tradeoff as a function of total compute

These represent a different class of research from hyperparameter sweeps and are the natural next step for the non-record track.

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
