# Depth Recurrence + Legal Score-First TTT with SGD Momentum

**val_bpb = 1.14458** | Pre-TTT: 1.1611 | TTT gain: **−0.0165** | Artifact: 14.79 MB

> Non-record unlimited-compute submission (trained on 4×A100-40GB, eval 1046s on 1×A100).

---

## Headline Result

This submission demonstrates that **competition-legal test-time training (TTT) can deliver large gains** (−0.0165 BPB) when the TTT recipe is properly tuned.  The key insight is that SGD with momentum, applied for multiple epochs per chunk while freezing early layers, extracts **2.4× more TTT improvement** than single-epoch AdamW over the full network (−0.0068 in our prior submission, PR #456).

The final 1.1446 BPB comes entirely from a legal score-first protocol: every validation token is **scored before any weight update** that could use it, enforced by `torch.inference_mode()` during the scoring phase.

---

## Novel & Creative Contributions

### 1. High-Yield Legal TTT via Selective Freezing + SGD Momentum

Most TTT approaches in the competition use AdamW over all parameters and train for a single epoch.  We find a much more effective recipe:

- **SGD + momentum (0.9)** instead of AdamW — simpler optimizer with implicit regularization; lower memory footprint (no second-moment buffers) enables larger effective batch processing.
- **3 epochs per chunk** instead of 1 — repeated passes over each 32K-token chunk let the model fully adapt, especially on domain-specific or rare constructions.
- **Freeze the first 2 blocks** during TTT — early blocks learn general tokenization features (embeddings, basic syntax); adapting them hurts more than it helps. Freezing them regularizes TTT and keeps 19.9M of 24.6M parameters trainable on later, more "semantic" layers.

This combination yields a TTT gain of **−0.0165 BPB** (1.1611 → 1.1446), compared to −0.0068 with our prior AdamW-1-epoch approach.

### 2. Depth Recurrence (Weight-Efficient Deep Networks)

The model uses 11 logical layers but only **10 unique BlockCores** — one core is reused at two different depths. Each Block wraps a shared core with its own per-layer LayerNorm buffers and scaling factors, so the reused core sees different normalization statistics at each depth.

This delivers the representation capacity of an 11-layer network at the parameter cost of 10 layers — crucial in a size-constrained competition.  The technique is inspired by Universal Transformers but applied at the block level with independent normalization, avoiding the training instabilities of naive weight tying.

### 3. Partial Rotary Position Embeddings (16 of 64 dims)

Instead of applying RoPE to all head dimensions, we apply it to only the first 16 of 64 dimensions per head.  The remaining 48 dimensions are position-agnostic, acting as a "content-only" channel.

This has two benefits:
- **Better length generalization** — fewer dimensions are locked to absolute position, so the model degrades less gracefully on longer sequences during TTT.
- **NTK-aware scaling** — the 16 RoPE dimensions use dynamic NTK base scaling (`base * scale^(d/(d-2))`) for extended contexts, concentrating position information in a compact subspace.

### 4. Value Embeddings on Deep Layers Only

Layers 9 and 10 receive **128-dim learned value embeddings** — a separate embedding table whose output is added to the value projection before attention.  This gives deep layers direct access to token identity information in the value stream, bypassing the information bottleneck of the residual stream.

The embeddings are applied only to the deepest layers because:
- Early layers benefit more from positional/syntactic features than raw token identity.
- Adding VE everywhere wastes parameter budget (the 128-dim embedding table costs ~131K parameters).
- Per-layer scale factors (initialized to 0.1) allow the model to smoothly learn how much value-embedding signal to mix in.

### 5. Layer-Norm Depth Scaling

Each block's attention and MLP outputs are scaled by `1/√(layer_idx + 1)`, so deeper layers contribute smaller residual updates.  This stabilizes training for deeper networks under depth recurrence, where the same core processes inputs at multiple depths with different effective scales.

---

## Architecture Summary

| Component | Configuration |
|---|---|
| Layers | 11 logical (10 unique shared BlockCores) |
| Embedding dim | 768 |
| Heads | 12 (64 dim/head), 4 KV heads |
| MLP | 3× expansion (2304) with SwiGLU-style SmearGate |
| Vocab | 1024 (SentencePiece BPE) |
| BigramHash | 2048 features |
| RoPE | Partial: 16/64 dims, NTK-aware scaling |
| Value Embeddings | 128d on layers 9–10, per-layer scale (init 0.1) |
| LN Scale | `1/√(layer+1)` depth scaling |
| XSA | Cross-sequence attention on last 4 layers |
| U-Net skips | Residual connections across layer pairs |
| Parameters | 24,634,452 total |

## Training Details

| Setting | Value |
|---|---|
| Hardware | 4×A100-40GB (NVIDIA) |
| Steps | 5,200 |
| Training wallclock | 2,472s (~41 min) |
| Optimizer | Muon (hidden/attn) + Adam (embeddings/scalars) |
| SWA | 12 checkpoints from step 4,650 |
| Late QAT | Enabled at step 4,901 (scale < 0.1) |
| Quantization | Int6 + zstd-22 |

## TTT Protocol (Legal Score-First)

```
for each 32K-token chunk:
    1. model.eval() + torch.inference_mode()
       → Forward pass on chunk, accumulate NLL    ← SCORE (graded)
    2. model.train()
       → SGD(lr=0.002, momentum=0.9), 3 epochs   ← TRAIN (adaptation)
    3. Advance to next chunk with updated weights
```

Every target token is scored exactly once, strictly before any gradient update that could benefit from it.  The `torch.inference_mode()` context manager makes gradient leakage during scoring physically impossible.

| TTT Setting | Value |
|---|---|
| Optimizer | SGD, momentum=0.9 |
| Learning rate | 0.002 |
| Epochs per chunk | 3 |
| Chunk size | 32,768 tokens |
| Stride | 64 |
| Frozen blocks | First 2 (of 11) |
| Trainable params | 19,911,748 / 24,634,452 |
| Eval time | 1,046s (1×A100) |

## Quantization & Size

| Component | Bytes |
|---|---|
| Model (int6 + zstd) | 14,717,713 |
| Code (train_gpt.py) | 71,706 |
| **Total** | **14,789,419** |
| Limit | 16,000,000 |
| Headroom | 1,210,581 (7.6%) |

## Training Curve

| Step | Val BPB | Notes |
|---|---|---|
| 0 | 4.1037 | |
| 500 | 1.4046 | |
| 1000 | 1.3226 | |
| 1500 | 1.2947 | |
| 2000 | 1.2626 | |
| 2500 | 1.2425 | |
| 3000 | 1.2265 | |
| 3500 | 1.2123 | |
| 4000 | 1.1982 | |
| 4500 | 1.1821 | |
| 5000 | 1.1654 | SWA started at 4650, Late QAT at 4901 |
| **5200** | **1.1611** | Pre-TTT baseline |
| **TTT** | **1.14458** | −0.0165 from legal score-first TTT |

## Comparison to Prior Submission (PR #456)

| Metric | PR #456 (10L) | This (11L) | Δ |
|---|---|---|---|
| **val_bpb** | 1.15321 | **1.14458** | **−0.00863** |
| Pre-TTT BPB | 1.1600 | 1.1611 | +0.0011 |
| TTT gain | −0.0068 | **−0.0165** | **2.4× larger** |
| Layers | 10 | 11 (10 unique) | +1 |
| BigramHash | 10240 | 2048 | −8192 |
| Artifact size | 15.98 MB | 14.79 MB | −1.19 MB |

The pre-TTT baselines are nearly identical (1.1600 vs 1.1611).  The entire improvement comes from better TTT — validating that the SGD+momentum + freeze + multi-epoch recipe is the key advance.

## Reproducibility

```bash
# Environment: Python 3.10+, PyTorch 2.x with CUDA
# From the repo root:
RUN_ID=i15_11L_ve128 \
NUM_LAYERS=11 \
UNIQUE_LAYERS=10 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=5200 \
VAL_LOSS_EVERY=500 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
ROPE_DIMS=16 LN_SCALE=1 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 EVAL_STRIDE=64 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 \
TTT_FREEZE_BLOCKS=2 TTT_BATCH_SEQS=32 TTT_MOMENTUM=0.9 \
torchrun --standalone --nproc_per_node=4 \
   records/track_non_record_16mb/2026-03-22_11L_VE128_PartialRoPE_LegalTTT/train_gpt.py
```
