# Depth Recurrence + Legal Score-First TTT with SGD Momentum (30 Epochs)

**val_bpb = 1.14252** | Pre-TTT: 1.1609 | TTT gain: **−0.0184** | Artifact: 15.48 MB

> Non-record unlimited-compute submission (trained on 4×A100-40GB, eval 3662s on 1×A100).

---

## Headline Result

This submission demonstrates that **competition-legal test-time training (TTT) can deliver very large gains** (−0.0184 BPB) when the TTT recipe is properly tuned.  The key discovery is that SGD with momentum applied for **30 epochs per chunk** while freezing early layers extracts **2.7× more TTT improvement** than our prior 1-epoch AdamW approach (−0.0068 in PR #456), and **12% more** than the 3-epoch SGD baseline (−0.0165 in PR #461).

The final 1.1425 BPB comes entirely from a legal score-first protocol: every validation token is **scored before any weight update** that could use it, enforced by `torch.inference_mode()` during the scoring phase.

---

## Novel & Creative Contributions

### 1. High-Yield Legal TTT via Selective Freezing + SGD Momentum

Most TTT approaches in the competition use AdamW over all parameters and train for a single epoch.  We find a much more effective recipe:

- **SGD + momentum (0.9)** instead of AdamW — simpler optimizer with implicit regularization; lower memory footprint (no second-moment buffers) enables larger effective batch processing. Our experiments confirm SGD outperforms AdamW by 0.027 BPB for legal TTT because Adam's moment estimates cannot converge with only ~30 optimization steps per chunk.
- **30 epochs per chunk** instead of 1 — repeated passes over each 32K-token chunk let the model fully adapt, especially on domain-specific or rare constructions. A sweep from 3→30 epochs showed general improvement, though not strictly monotonic at every point (see table below).
- **Freeze the first 2 blocks** during TTT — early blocks learn general tokenization features (embeddings, basic syntax); adapting them hurts more than it helps. Freezing them regularizes TTT and keeps 19.9M of 24.6M parameters trainable on later, more "semantic" layers.

This combination yields a TTT gain of **−0.0184 BPB** (1.1609 → 1.1425), compared to −0.0068 with our prior AdamW 1-epoch approach (PR #456) and −0.0165 with our 3-epoch SGD approach (PR #461).

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
| Embedding dim | 512 |
| Heads | 8 (64 dim/head), 4 KV heads (GQA) |
| MLP | 3× expansion (1536), ReLU² activation |
| SmearGate | Learned token-mixing gate on input embeddings |
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
| Training wallclock | 2,455s (~41 min) |
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
       → SGD(lr=0.002, momentum=0.9), 30 epochs  ← TRAIN (adaptation)
    3. Advance to next chunk with updated weights
```

Every target token is scored exactly once, strictly before any gradient update that could benefit from it.  The `torch.inference_mode()` context manager makes gradient leakage during scoring physically impossible.

| TTT Setting | Value |
|---|---|
| Optimizer | SGD, momentum=0.9 |
| Learning rate | 0.002 |
| Epochs per chunk | 30 |
| Chunk size | 32,768 tokens |
| Stride | 64 |
| Frozen blocks | First 2 (of 11) |
| Trainable params | 19,911,748 / 24,634,452 |
| Eval time | 3,662s (1×A100) |

## Quantization & Size

| Component | Bytes |
|---|---|
| Model (int6 + zstd) | 15,408,253 |
| Code (train_gpt.py) | 71,739 |
| **Total** | **15,479,992** |
| Limit | 16,000,000 |
| Headroom | 520,008 (3.3%) |

## Training Curve

| Step | Val BPB | Notes |
|---|---|---|
| 0 | 4.1037 | |
| 500 | 1.4063 | |
| 1000 | 1.3232 | |
| 1500 | 1.2947 | |
| 2000 | 1.2620 | |
| 2500 | 1.2424 | |
| 3000 | 1.2262 | |
| 3500 | 1.2122 | |
| 4000 | 1.1978 | |
| 4500 | 1.1819 | |
| 5000 | 1.1652 | SWA started at 4650, Late QAT at 4901 |
| **5200** | **1.1609** | Pre-TTT baseline |
| **TTT** | **1.14252** | −0.0184 from legal score-first TTT (30 epochs) |

## Comparison to Prior Submissions

| Metric | PR #456 (10L, 1ep AdamW) | PR #461 (11L, 3ep SGD) | This (11L, 30ep SGD) | Δ vs #461 |
|---|---|---|---|---|
| **val_bpb** | 1.15321 | 1.14458 | **1.14252** | **−0.00206** |
| Pre-TTT BPB | 1.1600 | 1.1611 | 1.1609 | −0.0002 |
| TTT gain | −0.0068 | −0.0165 | **−0.0184** | |
| TTT epochs | 1 | 3 | **30** | 10× |
| Eval time | 356s | 1,046s | 3,662s | 3.5× |
| Artifact size | 15.98 MB | 14.79 MB | 15.48 MB | +0.69 MB |

The pre-TTT baselines are nearly identical (1.1600 → 1.1611 → 1.1609).  The entire improvement comes from more TTT epochs — a sweep from 3→30 epochs showed general improvement with some non-monotonicity around 15 epochs: 

### TTT Epoch Sweep Results (SGD, freeze=2, lr=0.002)

| Epochs | BPB | Δ vs 3ep | Notes |
|---|---|---|---|
| 3 | 1.14458 | baseline | PR #461 |
| 5 | 1.14399 | −0.00059 | |
| 7 | 1.14378 | −0.00080 | |
| 10 | 1.14295 | −0.00163 | |
| 15 | 1.14335 | −0.00123 | Non-monotonic (worse than 10ep) |
| 20 | 1.14292 | −0.00166 | |
| **30** | **1.14252** | **−0.00206** | This submission |

All results are single runs (no error bars). The non-monotonicity at 15 epochs suggests some variance; further runs would be needed to establish statistical significance of ordering among nearby epoch counts.

40-epoch and 50-epoch runs are in progress at time of submission.

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
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=30 \
TTT_FREEZE_BLOCKS=2 TTT_BATCH_SEQS=32 TTT_MOMENTUM=0.9 \
torchrun --standalone --nproc_per_node=4 \
   records/track_non_record_16mb/2026-03-23_11L_VE128_PartialRoPE_LegalTTT_30ep/train_gpt.py
```

## Credits

This submission builds on work from many contributors to the parameter-golf competition:

- **Muon optimizer** — Baseline (`modded-nanogpt`); Newton-Schulz orthogonal preconditioning
- **BigramHash embeddings** — PR #65 (aquariouseworkman): hash consecutive token pairs for cheap bigram context
- **SmearGate** — PR #65 (aquariouseworkman): per-dim sigmoid gate blending adjacent token embeddings
- **XSA (Exclusive Self Attention)** — PR #187 (Idan3011): removes self-value bias via orthogonal projection; GQA-aware variant in PR #265 (unnir)
- **U-Net skip connections** — PR #65 (aquariouseworkman), PR #69 (TevBenji): encoder-decoder layer pairing with learned skip weights
- **Mixed int5/int6 quantization** — PR #76 (unixmadtoonslab / Will DePue): int5 for MLP, int6 for attention
- **SWA (Stochastic Weight Averaging)** — PR #69 (TevBenji): checkpoint averaging during warmdown
- **Late QAT** — PR #315 (jfprincz), working implementation in PR #374 (unnir): STE fake-quantization in final training phase
- **Sliding window evaluation** — PR #50 (mattqlf / Matthew Li): stride-64 overlapping windows
- **Value Embeddings (VE128)** — PR #374 (unnir): learned embeddings added to value projections on deep layers
- **Partial RoPE (16/64 dims)** — PR #315 (jfprincz), PR #374 (unnir): rotary embeddings on 25% of head dims
- **LN Scale (depth scaling)** — PR #315 (jfprincz), PR #374 (unnir): `1/√(layer+1)` output scaling
- **Legal TTT framework** — PR #77 (samacqua): first legal score-first TTT (LoRA); full-model variant in our PR #456
- **Score-first protocol + SGD TTT** — Our prior work (PR #461): `torch.inference_mode()` scoring, SGD+momentum, freeze-2
- **ReLU² activation, GQA** — Baseline (`modded-nanogpt`)

Built on the [parameter-golf](https://github.com/openai/parameter-golf) starter code by Beren Millidge & Keller Jordan.
