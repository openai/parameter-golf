# 11L GEPA + 12k Steps + Pure Int6 + 15-Percentile GPTQ-lite + Legal TTT

**val_bpb = 1.1079** | Pre-TTT float: 1.1268 | Int6 quant: ~1.154 | TTT gain: **−0.046** | Artifact: 14.72 MB | Submission: 14.79 MB

> Non-record unlimited-compute submission (trained on 4×A100-40GB, eval 2072s on 4×A100).

---

## Headline Result

This submission demonstrates that **extended training (12000 steps)** combined with **pure int6 quantization** and **legal score-first TTT** can push BPP substantially below the current leaderboard #1 (signalrush: 1.1228). The key insight is that more training steps at peak learning rate — 7000 steps before warmdown — produces a significantly better float base model (1.1268 vs 1.1418 for signalrush at 7051 steps), and the improved GPTQ-lite quantization with 15-candidate clip search keeps the quantization gap small.

The training recipe exploits the non-record track's unlimited compute allowance to train 70% longer than the standard 10-minute window would permit, yielding a **1.1079 BPB** result that beats the current SOTA by **0.0149 BPP**.

---

## Novel & Creative Contributions

### 1. 12000-Step Training with 5000-Step Warmdown

Training for 12000 steps (vs the standard 5200 or signalrush's 7051) with a 5000-step cosine warmdown starting at step 7000. This gives 7000 steps at peak learning rate for weight optimization, followed by a gentle 5000-step anneal. The longer peak-LR phase produces a float base of **1.1268** — 0.015 lower than signalrush's 1.1418.

Diminishing returns observed:
- 5200→7000 steps: −0.007/kstep
- 7000→9000 steps: −0.006/kstep
- 9000→12000 steps: −0.003/kstep

### 2. Pure Int6 Per-Row Quantization with 15-Candidate GPTQ-lite

All model tensors (including embeddings) use int6 per-row quantization with GPTQ-lite optimal clip search. The upgrade from 5 to **15 clip percentiles** improves the clip point optimization, reducing quantization error and producing artifacts ~237KB smaller than the 5-percentile version. Combined with zstd-22 compression, the 27M-parameter model compresses to just **14.72 MB** of model data.

The pure int6 approach (vs mixed int6/int8) produces smaller artifacts because:
- No tensor-type routing overhead
- Uniform per-row scale storage (float16)
- Better zstd compression due to uniform bit patterns

### 3. Legal Score-First TTT with SGD Momentum + LR Warmup

Competition-legal test-time training using sliding windows with score-first protocol — every token is scored under `torch.inference_mode()` before any weight update. Key improvements over baseline TTT:

- **LR warmup**: Linear warmup over the first 5% of chunks before cosine decay, reducing early over-adaptation
- **SGD with momentum** (0.9) at lr=0.002 for 10 epochs per 32K-token chunk
- **Freeze first 2 blocks**: Prevents catastrophic forgetting in early layers
- **Gradient clipping** at 1.0 for stability

Yields **−0.046 BPP** improvement (1.154 → 1.108).

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
| Steps | 12,000 |
| Warmdown | Cosine anneal from step 7,000 to 12,000 |
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
| Model bytes | 15,432,359 |
| Code bytes | 78,281 |
| **Total submission** | **15,510,640 bytes (14.79 MB)** |

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
| Eval time | 2,072 seconds |

## Training Trajectory

| Step | val_bpb | Phase |
|---|---|---|
| 0 | 4.104 | Init |
| 500 | 1.393 | Warmup complete |
| 1000 | 1.324 | Peak LR |
| 2000 | 1.265 | Peak LR |
| 3000 | 1.245 | Peak LR |
| 5000 | 1.230 | Peak LR |
| 7000 | 1.226 | Warmdown start |
| 8000 | 1.205 | Warmdown |
| 9000 | 1.196 | Warmdown |
| 10000 | 1.178 | Warmdown |
| 11000 | 1.155 | Warmdown |
| 11500 | 1.140 | Warmdown |
| 12000 | **1.127** | **Final float** |

## Comparison with Prior Results

| Run | Steps | Float | Quant | TTT BPP | Artifact | Status |
|---|---|---|---|---|---|---|
| **This (exp_12k_pur)** | **12000** | **1.127** | **~1.154** | **1.1079** | **14.79 MB** | **✅ NEW BEST** |
| exp_9k_pure | 9000 | 1.135 | ~1.162 | 1.1157 | 14.94 MB | ✅ |
| gep_9k3k | 9000 | 1.136 | ~1.162 | 1.1153 | 15.97 MB | ✅ |
| gep_v27k | 7000 | 1.148 | ~1.174 | 1.1334 | 15.63 MB | ✅ |
| signalrush (SOTA) | 7051 | 1.142 | 1.149 | 1.1228 | 15.55 MB | Record track |
