# SP1024 + Depth Recurrence + Adaptive Markov Curriculum + Auto-QMax GPTQ + Legal TTT

**val_bpb: 1.1047** (single seed) | **15.89 MB** | 8×H100 SXM

---

*A quick personal note before the technical details:*

*Being part of this challenge and putting up a meaningful score meant a lot to me. We were supposed to go on vacation next month — my fiancée Virginia and I — but I spent that budget on H100 runs instead. I don't come from an ML lab, I'm trying to learn and keep going on my own. But having her support through this process meant everything — still sitting next to me at 3 AM saying "keep going" is something I won't forget.*

---

## Results (8×H100 80GB SXM, SEED=42)

| Metric | Value |
|--------|-------|
| Training steps | 5,183 (wallclock cap at 600s) |
| Pre-quant val_bpb | 1.1359 |
| Post-quant val_bpb | 1.1429 |
| Sliding window val_bpb | 1.1065 |
| **TTT final val_bpb** | **1.1047** |
| Artifact size | 15,888,861 bytes |
| Training time | 590s |
| Eval time (sliding + TTT) | ~491s |

## Approach

This submission combines techniques from several existing PRs with three original contributions. The core insight: most submissions treat quantization as a post-training afterthought, but the interplay between model capacity, clip range, and compressed size is the binding constraint of this challenge.

### Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 heads, 4 KV heads) |
| MLP | 4× with LeakyReLU(0.5)² |
| Depth Recurrence | Layers 3,4,5 repeated once → 14 effective layers |
| Parallel Residuals | From layer 7, merged via learned gate |
| XSA | All 11 layers |
| Partial RoPE | 16 of 64 head dims |
| LN Scale | 1/√(layer+1) |
| BigramHash | 1,536 buckets, 128d |
| Value Embedding | 128d, layers 9–10 |
| Skip Gates | Learned sigmoid gating on U-Net connections |
| Logit Softcap | 30.0 |

### Training

| Parameter | Value |
|-----------|-------|
| Optimizer (matrices) | Muon + MuonEq-R + AOL preconditioning |
| Matrix LR / WD | 0.022 / 0.095 |
| Muon momentum | 0.99 (warmup from 0.92 over 1,500 steps) |
| Embedding LR | 0.03 (tied) |
| Batch tokens | 786,432 |
| Sequence length | 1,024 |
| Recurrence activation | Step 2,000 |
| Late QAT | Last 200 steps, int6 STE + `_dynamo.reset()` |
| Weight averaging | 30% EMA(0.997) + 70% SWA(start=75%) |

### Original Contributions

**1. Adaptive Markov Curriculum**

Bigram-surprise-weighted loss scaling. A bigram transition table is built from training data at initialization. Each batch receives a loss multiplier based on how surprising its token transitions are to the bigram model — tokens the bigram already predicts well get baseline weight, tokens with high bigram surprise get up to 10% extra. This steers capacity toward patterns that n-gram statistics can't capture.

**2. Auto-QMax Binary Search**

Binary search over [31, 127] to find the maximum int6 clip range whose compressed artifact fits under 16 MB. For this model (32.7M params, MLP 4×) it lands at qmax=41. In earlier experiments with smaller models, this reduced quantization gap from 0.032 to 0.008 BPB — the difference between a 11.5 MB artifact wasting 4.5 MB of budget and actually using it.

The realization that drove this: a model at qmax=71 / 15.9 MB always beats the same model at qmax=31 / 11.5 MB. Leaving megabytes on the table is leaving BPB on the table.

**3. Turbo-Muon with AOL Diagonal Preconditioning**

Row-normalized Muon (MuonEq-R) extended with diagonal preconditioning: `D_r = diag(UU^T)^{1/2}`, `D_c = diag(U^TU)^{1/2}`, applied before Newton-Schulz. This balances gradient magnitudes across both dimensions, stabilizing convergence under the aggressive WD=0.095 needed for quantization-friendly weight distributions.

### Quantization & Compression

| Component | Method |
|-----------|--------|
| Matrix weights | Int6 GPTQ (Hessian + Cholesky + actorder) |
| Embeddings | Int8 per-row |
| Clip range | Auto-searched (qmax=41) |
| Compression | Brotli-11 + byte-shuffle |
| Budget fitting | Selective ±1 pruning |

### Legal Score-First TTT

Each 32K-token chunk is scored under `torch.inference_mode()` first, then used for 3 epochs of SGD adaptation (lr=0.002, momentum=0.9). Every token is graded before any weight update that could benefit from it. The last chunk is scored but never trained on.

## Run Command

```bash
QK_GAIN_INIT=5.0 MIN_LR=0.05 \
RECUR_LAYERS=3,4,5 RECUR_START_STEP=2000 \
PARALLEL_START_LAYER=7 \
MUON_WD=0.095 MATRIX_LR=0.022 RAKI_POWER=0.10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 \
SWA_ENABLED=1 SWA_START_FRAC=0.75 \
BIGRAM_ENABLED=1 BIGRAM_VOCAB=1536 BIGRAM_DIM=128 \
LATE_QAT=1 GPTQ_ENABLED=1 \
MAX_WALLCLOCK_SECONDS=600 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credit

Built on work from PR #1339 (@bigbag), PR #549 (@abaybektursun), PR #287 and #198 (@jfprincz), PR #374 (@signalrush).

And Virginia.
