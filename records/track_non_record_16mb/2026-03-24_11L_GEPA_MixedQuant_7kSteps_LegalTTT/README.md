# 11L GEPA + Mixed Int6/Int8 Quantization + 7k Steps + Legal Score-First TTT

**val_bpb = 1.1334** | Pre-TTT: 1.1476 | TTT gain: **−0.0142** | Artifact: 15.70 MB

> Non-record unlimited-compute submission (trained on 4×A100-40GB, eval 2194s on 4×A100).

---

## Headline Result

This submission pushes base model quality by training for **7000 steps** (35% more than the standard 5200) using a **GEPA architecture** (SwiGLU + U-Net + BigramHash + EMA + XSA + GPTQ-lite) with **mixed int6/int8 quantization** to stay under the 16MB cap. The longer training brings the pre-TTT baseline down to **1.1476 BPB**, and legal score-first TTT with SGD momentum (10 epochs) yields a further **−0.0142 BPB** improvement to reach **1.1334 BPB**.

The mixed quantization strategy — int6 per-row with GPTQ-lite clip search for the large QAT-trained attention and MLP weights, int8 per-tensor scalar for the rest — compresses the 27M-parameter model into a 15.70 MB artifact with minimal accuracy loss.

---

## Novel & Creative Contributions

### 1. Extended Training (7000 Steps) for Deeper Loss Basin

Training 35% longer than the standard 5200 steps, with warmdown starting at step 3500, gives the model more time to converge. The extra steps particularly help during the warmdown cosine anneal from step 3500 to 7000, allowing gentler learning rate decay. Base model improves from 1.1570 BPB (5200 steps, prior submissions) to **1.1476 BPB** (7000 steps).

The key enabler is mixed quantization: the int6+int8 scheme compresses the model enough that even with a larger 27M-parameter model trained for 7000 steps, the artifact stays under 16MB.

### 2. Mixed Int6/Int8 Quantization for Size-Aware Compression

A dual-scheme quantization strategy:

- **QAT-trained attention and MLP weights** (the bulk of parameters) use **int6 per-row quantization** with GPTQ-lite clip search (5 candidates per row), minimizing per-row MSE.
- **Smaller tensors** (layer norms, value embeddings, biases, embedding tables) use **int8 per-tensor scalar quantization**, preserving accuracy in the tensors most sensitive to low-precision quantization.

This preserves accuracy where it matters most while keeping the large tensors compact. The mixed scheme compresses 27M parameters into 15.63 MB of model data (+ 76 KB code = 15.70 MB total).

### 3. GEPA Architecture (SwiGLU + U-Net + BigramHash + EMA + XSA + GPTQ-lite)

Combines several proven techniques into a coherent architecture:

- **ReLU² activation** in 3× MLP (1536 hidden)
- **Cross-sequence attention (XSA)** on the last 4 layers — removes self-value bias via orthogonal projection
- **Exponential moving average** (decay 0.997) applied every step
- **Bigram hash embeddings** (2048 buckets, 128-dim) — cheap bigram context via hash of consecutive token pairs
- **Partial RoPE** (16 of 64 dims) with YARN scaling — concentrates position info in a compact subspace
- **Late QAT** with 5-candidate GPTQ-lite clip search, triggered when LR scale drops below 0.15
- **Value embeddings** (128d) on layers 9–10 — direct token identity signal in the value stream for deep layers
- **U-Net skip connections** across layer pairs
- **LN depth scaling** (`1/√(layer+1)`) for stable deep training

### 4. Legal Score-First TTT with SGD Momentum

Competition-legal test-time training using sliding windows with score-first protocol — every token is scored under `torch.inference_mode()` before any weight update. SGD with momentum (0.9) at lr=0.002 for 10 epochs per 32K-token chunk, freezing the first 2 blocks. Yields **−0.0142 BPB** gain (1.1476 → 1.1334).

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
| RoPE | Partial: 16/64 dims, YARN scaling (train_seq=1024) |
| Value Embeddings | 128d on layers 9–10, per-layer scale (init 0.1) |
| LN Scale | `1/√(layer+1)` depth scaling |
| XSA | Cross-sequence attention on last 4 layers |
| U-Net skips | Residual connections across layer pairs |
| SmearGate | Learned token-mixing gate on input embeddings |
| Tied Embeddings | Yes |
| Parameters | 27,030,108 total |

## Training Details

| Setting | Value |
|---|---|
| Hardware | 4×A100-40GB (NVIDIA) |
| Steps | 7,000 |
| Warmdown | Cosine anneal from step 3,500 to 7,000 |
| Warmup | 20 steps |
| Training wallclock | 3,490s (~58 min) |
| Batch tokens | 786,432 |
| Sequence length | 2,048 |
| Optimizer | Muon (hidden/attn) + Adam (embeddings/scalars) |
| Muon WD | 0.04 |
| Adam WD | 0.04 |
| Decoder LR mult | 2.0 |
| Grad clip | 0.3 |
| EMA | Decay 0.997, every step |
| Late QAT | Enabled at step 6,476 (scale < 0.15) |

## TTT Protocol (Legal Score-First)

```
for each 32K-token chunk:
    1. model.eval() + torch.inference_mode()
       → Forward pass on chunk, accumulate NLL    ← SCORE (graded)
    2. model.train()
       → SGD(lr=0.002, momentum=0.9), 10 epochs  ← TRAIN (adaptation)
    3. Advance to next chunk with updated weights
```

Every target token is scored exactly once, strictly before any gradient update that could benefit from it. The `torch.inference_mode()` context manager makes gradient leakage during scoring physically impossible.

| TTT Setting | Value |
|---|---|
| Optimizer | SGD, momentum=0.9 |
| Learning rate | 0.002 |
| Epochs per chunk | 10 |
| Chunk size | 32,768 tokens |
| Stride | 64 |
| Frozen blocks | First 2 (of 11) |
| Trainable params | 22,301,260 / 27,030,108 |
| Eval time | 2,194s (4×A100) |

## Quantization & Size

| Component | Bytes |
|---|---|
| Model (mixed int6/int8 + zstd-22) | 15,626,769 |
| Code (train_gpt.py) | 76,429 |
| **Total** | **15,703,198** |
| Limit | 16,000,000 |
| Headroom | 296,802 (1.9%) |

Mixed quantization breakdown:
- **Int6 per-row** (GPTQ-lite, 5 clip candidates): attention projections, MLP weights — all QAT-trained tensors
- **Int8 per-tensor** (scalar scale): layer norms, value embeddings, biases, embedding tables
- **Payload**: 27,522,422 bytes → **zstd-22**: 15,626,769 bytes (3.89× compression)

## Training Curve

| Step | Val BPB | Notes |
|---|---|---|
| 0 | 4.1044 | |
| 500 | 1.4108 | |
| 1000 | 1.3334 | |
| 1500 | 1.3050 | |
| 2000 | 1.2711 | |
| 2500 | 1.2592 | |
| 3000 | 1.2509 | |
| 3500 | 1.2475 | Warmdown begins |
| 4000 | 1.2354 | |
| 4500 | 1.2231 | |
| 5000 | 1.2104 | |
| 5500 | 1.1970 | |
| 6000 | 1.1834 | |
| 6500 | 1.1642 | Late QAT enabled at 6476 |
| **7000** | **1.1476** | Pre-TTT baseline (EMA applied) |
| **TTT** | **1.1334** | −0.0142 from legal score-first TTT (10 epochs) |

## Comparison to Prior Submissions

| Submission | BPB | Status |
|---|---|---|
| **This work** | **1.1334** | Non-record |
| Prior: VE128+PartialRoPE+LegalTTT 30ep | 1.1425 | Non-record |
| Prior: VE128+PartialRoPE+LegalTTT 10ep | 1.1451 | Non-record |
| Record SOTA (signalrush) | 1.1228 | Record |

Key improvements over the prior VE128+PartialRoPE submission (1.1425):
- **Better base model**: 1.1476 vs 1.1609 (−0.0133) from 7000 steps + GEPA architecture
- **Fewer TTT epochs needed**: 10 epochs vs 30 epochs, yet still reaches a better final BPB
- **Faster eval**: 2,194s vs 3,662s (40% faster, partly from fewer TTT epochs and 4-GPU eval)

## Reproducibility

```bash
# Environment: Python 3.10+, PyTorch 2.x with CUDA
# From the repo root:
RUN_ID=gep_v27k \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=7000 \
WARMDOWN_ITERS=3500 \
VAL_LOSS_EVERY=500 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_HIDDEN=1536 \
TIE_EMBEDDINGS=1 \
BIGRAM_BUCKETS=2048 \
BIGRAM_EMBED_DIM=128 \
ROPE_DIMS=16 \
ROPE_TRAIN_SEQ_LEN=1024 \
LN_SCALE=1 \
XSA_LAYERS=4 \
EVAL_STRIDE=64 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
LATE_QAT=1 QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
GRAD_CLIP_NORM=0.3 \
MIXED_QUANT=1 QUANT_EMBED=1 \
TTT_ENABLED=1 TTT_OPTIMIZER=sgd \
TTT_LR=0.002 TTT_EPOCHS=10 \
TTT_FREEZE_BLOCKS=2 TTT_BATCH_SEQS=32 \
TTT_CHUNK_TOKENS=32768 TTT_MOMENTUM=0.9 \
torchrun --standalone --nproc_per_node=4 \
   records/track_non_record_16mb/2026-03-24_11L_GEPA_MixedQuant_7kSteps_LegalTTT/train_gpt.py
```

## Credits

This submission builds on work from many contributors to the parameter-golf competition:

- **Muon optimizer** — Baseline (`modded-nanogpt`); Newton-Schulz orthogonal preconditioning
- **BigramHash embeddings** — PR #65 (aquariouseworkman): hash consecutive token pairs for cheap bigram context
- **SmearGate** — PR #65 (aquariouseworkman): per-dim sigmoid gate blending adjacent token embeddings
- **XSA (Exclusive Self Attention)** — PR #187 (Idan3011): removes self-value bias via orthogonal projection; GQA-aware variant in PR #265 (unnir)
- **Value Embeddings** — Per-layer learned embeddings added to the value stream
- **GPTQ-lite clip search** — Per-row optimal clip percentile search for int6 quantization
