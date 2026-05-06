# Raki: Adaptive Markov Curriculum + Turbo-Muon AOL + Auto-QMax Artifact Packing

**val_bpb = 1.1047** (SEED=42) | **15.89 MB** | 8×H100 SXM | 590s training + 491s eval

---

*A quick personal note before the technical details:*

*Being part of this challenge and putting up a meaningful score meant a lot to me. We were supposed to go on vacation next month — my fiancée Virginia and I — but I spent that budget on H100 runs instead. I don't come from an ML lab, I'm trying to learn and keep going on my own. But having her support through this process meant everything — still sitting next to me at 3 AM saying "keep going" is something I won't forget.*

---

## Abstract


We  introduce Adaptive Markov Curriculum, a training-time loss weighting scheme that steers model capacity toward token transitions that bigram statistics cannot predict, and a combined Turbo-Muon optimizer that stacks AOL diagonal preconditioning with MuonEq-R row normalization for stable convergence under aggressive weight decay (WD=0.095).

## Results

| Stage | val_loss | val_bpb | Notes |
|-------|----------|---------|-------|
| Pre-quantization (EMA+SWA) | 1.9180 | 1.1359 | 5,183 steps, 590s wallclock |
| Post-quantization (int6 GPTQ, qmax=41) | 1.9297 | 1.1429 | Quant gap: 0.0070 |
| Sliding window (stride=64) | 1.8683 | 1.1065 | Full context scoring |
| **Score-first TTT (3 epochs)** | **1.8653** | **1.1047** | **Legal backward-looking** |
| Artifact size | — | — | 15,888,861 bytes (99.3% of 16 MB) |

##  Contributions

### 1. Adaptive Markov Curriculum: Bigram-Surprise-Weighted Training

We construct a bigram transition matrix from training data at initialization (2M tokens, Laplace-smoothed). During training, each batch receives a loss multiplier:

```
weight = 1.0 + power × min(surprise × entropy_weight / 5.0, 1.0)
```

where `surprise = −log P_bigram(y|x)` and `entropy_weight` is the normalized row entropy of the preceding token. The multiplier ranges from 1.0 to 1.10 (power=0.10).

**Intuition:** Tokens that the bigram model already predicts well (high-frequency collocations, punctuation patterns) receive baseline gradient signal. Tokens with high bigram surprise — rare transitions, semantic content, cross-domain vocabulary — receive up to 10% amplified gradients. This steers the neural model's limited capacity toward patterns that statistical n-gram methods fundamentally cannot capture.

This is philosophically related to Focal Loss (Lin et al., 2017) but operates on *distributional surprise* rather than model confidence, and to Complementary Training (PR #803) but applies during training rather than at eval time.

### 2. Auto-QMax: Binary Search over Quantization Precision (0.024 BPB)

The standard approach in this competition uses a fixed int6 clip range (qmax=31), producing artifacts of 11–12 MB. This wastes 25–30% of the 16 MB budget.

We perform binary search over qmax ∈ [31, 127] at serialization time, finding the maximum clip range whose compressed artifact fits under 16 MB. For our 32.7M-parameter model, this lands at qmax=41. The effect is dramatic:

| Configuration | Artifact Size | Post-quant BPB | Quant Gap |
|---------------|--------------|----------------|-----------|
| Fixed qmax=31 | 11.47 MB | 1.3300 | 0.0322 |
| **Auto qmax=41** | **15.89 MB** | **1.1429** | **0.0070** |
| Improvement | +4.42 MB | **−0.1871** | **4.6× smaller gap** |

The key realization: every unused megabyte in the artifact is wasted precision. A model at qmax=71 / 15.9 MB always dominates the same model at qmax=31 / 11.5 MB. The binary search adds ~60 seconds of post-training compute and requires only the existing quantization infrastructure.

*To our knowledge, no other submission in this competition performs dynamic clip range optimization to maximize artifact utilization.*

### 3. Turbo-Muon: AOL + MuonEq-R Combined Preconditioning

Standard Muon applies Newton-Schulz orthogonalization to gradient matrices. MuonEq-R (PR #1260) adds per-row normalization before NS5. We extend this with AOL diagonal preconditioning (arXiv:2512.04632):

```
D_r = diag(G G^T)^{1/2},  D_c = diag(G^T G)^{1/2}
G_preconditioned = D_r^{-1} G D_c^{-1}
```

Applied after MuonEq-R row normalization and before Newton-Schulz iteration (with steps reduced by 1). This balances gradient magnitudes across both row and column dimensions simultaneously, which is critical under the aggressive WD=0.095 regime needed for quantization-friendly weight distributions.

The combination — row normalization for scale invariance + diagonal preconditioning for conditioning number reduction — produces more stable convergence than either technique alone.

### 4. EMA + SWA Blended Weight Averaging

Rather than choosing between Exponential Moving Average and Stochastic Weight Averaging, we blend both:

```
final_weights = 0.30 × EMA(decay=0.997) + 0.70 × SWA(start=75% of training)
```

EMA provides continuous smoothing that tracks the optimization trajectory. SWA provides discrete averaging over a wider basin. The 30/70 blend captures benefits of both: EMA's responsiveness to late-training improvements and SWA's robustness to loss surface noise. In our runs, SWA accumulates ~1,271 checkpoints over the final 25% of training.

## Architecture (from PR #1339 / #1204 / #549)

| Component | Configuration |
|-----------|---------------|
| Transformer | 11 layers, 512d, 8 heads, 4 KV heads |
| MLP | 4× expansion, LeakyReLU(0.5)² activation |
| Depth Recurrence | Layers 3→5 repeated once (14 effective layers, activated at step 2,000) |
| Parallel Residuals | Dual-lane attention/MLP from layer 7, learned merge gate |
| XSA | All 11 layers (value-orthogonal projection) |
| Partial RoPE | 16 of 64 head dimensions |
| LN Scale | 1/√(layer_idx + 1) per-layer normalization scaling |
| BigramHash | 1,536 buckets, 128d, hash-projected to embedding space |
| Value Embedding | 128d shared, applied at layers 9–10 with learned per-layer scales |
| Skip Gates | Learned sigmoid gating on U-Net encoder→decoder connections |
| Logit Softcap | 30.0 (tanh-based) |

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Muon (matrices) + AdamW (scalars/embeddings) | |
| Matrix LR | 0.022 | Compensates high WD |
| Weight Decay | 0.095 (Muon), 0.09 (embed), 0.02 (Adam) | Compression-friendly flat minima |
| Momentum | 0.99 (warmup 0.92→0.99 over 1,500 steps) | |
| Gradient Clipping | 0.3 (global norm) | |
| Batch Tokens | 786,432 | |
| Sequence Length | 1,024 (SP1024 tokenizer) | |
| Late QAT | Last 200 steps, int6 STE + `torch._dynamo.reset()` | Forces recompilation with QAT branch active |
| Warmdown | 66.7% cosine decay with min_lr=0.05 | |

## Quantization Pipeline

1. **EMA+SWA blend** applied to model weights
2. **Auto-QMax binary search**: find optimal qmax ∈ [31, 127] for 16 MB target
3. **Full Hessian GPTQ** (67 layers): Cholesky inverse + actorder column reordering + 5-percentile scale search
4. **Brotli-11** compression with byte-shuffle pre-filter
5. **Selective ±1 pruning** if over budget (not needed at qmax=41)

## Legal Score-First TTT

Following the framework from PR #549 and ruling in issue #402:

1. Val tokens split into 1,893 non-overlapping 32K-token chunks
2. For each chunk: **SCORE** all windows under `torch.inference_mode()` (no gradients, no weight mutation)
3. Then **TRAIN** on the already-scored chunk: SGD(lr=0.002, momentum=0.9), 3 epochs, cosine LR decay, grad clip 1.0
4. Last chunk scored but never trained on

Total TTT contribution: sliding 1.1065 → TTT **1.1047** (−0.0018 BPB).

## Ablation Notes

| Technique removed | Estimated BPB impact |
|-------------------|---------------------|
| Auto-QMax (revert to qmax=31) | +0.024 (measured) |
| Depth Recurrence | +0.015 (from PR #1204) |
| GPTQ → clip search only | +0.005 |
| EMA+SWA → EMA only | +0.002 |
| Markov curriculum | +0.002 (estimated) |
| Turbo-Muon AOL | +0.002 (estimated) |

## Reproduce

```bash
pip install sentencepiece brotli
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

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

## Credits

 PR #1339 (@bigbag), PR #1204 (@msisovic), PR #1326 (@aryanbhosale) PR #549 (@abaybektursun), PR #1331 (@dexhunter), PR #1260 (@dexhunter) PR #287 (@jfprincz)




