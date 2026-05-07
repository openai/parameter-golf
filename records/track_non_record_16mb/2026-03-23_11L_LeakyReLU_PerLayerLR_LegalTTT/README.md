# Depth Recurrence + LeakyReLU(0.5)² + Per-Layer LR Legal TTT (30 Epochs)

**val_bpb = 1.13872** (best seed) | **3-seed mean: 1.13936 ± 0.0008** | Pre-TTT mean: 1.1574 | TTT gain: **−0.0182** | Artifact: 15.36 MB

> Non-record unlimited-compute submission (trained on 4×A100-40GB, eval ~3690s on 1×A100).

---

## Headline Result

This submission integrates three techniques from recent PRs (#518, #481) with our legal score-first TTT recipe, achieving **BPB 1.13872** — improving on our prior 1.14252 (PR #526) by **−0.0038 BPB**. Validated across 3 seeds for reproducibility.

| Seed | BPB | Δ vs PR #526 |
|------|-----|-------------|
| 1337 | 1.13912 | −0.00340 |
| 42 | 1.14024 | −0.00228 |
| **7** | **1.13872** | **−0.00380** |
| **Mean ± std** | **1.13936 ± 0.0008** | **−0.00316** |

---

## What Changed vs PR #526

### Architecture: LeakyReLU(0.5)² Activation (from PR #518)

Replace `relu(x)²` with `leaky_relu(x, 0.5)²` in the MLP. This preserves negative gradient flow, allowing the model to encode information in both positive and negative activations. The squaring still provides the non-linearity.

- Pre-TTT BPB improvement: 1.1609 → 1.1574 mean (**−0.0035** across 3 seeds)
- Zero compute overhead, same parameter count
- This accounts for essentially all of the final BPB improvement.

### TTT Recipe: Per-Layer LR + Intra-Chunk Cosine (from PR #481/#518)

Two modifications to the TTT recipe, adopted from other PRs:

1. **Per-layer LR groups**: `mlp.proj` gets 3× LR (higher quantization error), `mlp.fc` gets 0.5× LR.
2. **Intra-chunk cosine decay**: within each chunk's 30 TTT epochs, LR follows `0.5 × (1 + cos(π × step / total_steps))`.

However, the TTT gain actually went from −0.0184 (PR #526) to −0.0182 (this PR), a **+0.0002 regression**. Without ablations isolating per-layer LR and intra-cosine from the LeakyReLU architecture change, we cannot confirm these TTT modifications help. They may be neutral or slightly negative on this architecture.

---

## Architecture Summary

| Component | Configuration |
|---|---|
| Layers | 11 logical (10 unique shared BlockCores) |
| Embedding dim | 512 |
| Heads | 8 (64 dim/head), 4 KV heads (GQA) |
| MLP | 3× expansion (1536), **LeakyReLU(0.5)²** activation |
| SmearGate | Learned token-mixing gate on input embeddings |
| Vocab | 1024 (SentencePiece BPE) |
| BigramHash | 2048 features, 128d |
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
| Training wallclock | ~2,509s (~42 min) |
| Optimizer | Muon (hidden/attn) + Adam (embeddings/scalars) |
| SWA | Checkpoints from step 4,650 |
| Late QAT | Enabled at step 4,901 |
| Quantization | Int6 + zstd-22 |

## TTT Protocol (Legal Score-First + Per-Layer LR)

```
for each 32K-token chunk:
    1. model.eval() + torch.inference_mode()
       → Forward pass on chunk, accumulate NLL    ← SCORE (graded)
    2. model.train()
       → SGD(momentum=0.9), 30 epochs             ← TRAIN (adaptation)
         per-layer LR: mlp.proj 3x, mlp.fc 0.5x
         intra-chunk cosine LR decay
         inter-chunk cosine LR decay
    3. Advance to next chunk with updated weights
```

Every target token is scored exactly once, strictly before any gradient update that could benefit from it. The `torch.inference_mode()` context manager makes gradient leakage during scoring physically impossible.

| TTT Setting | Value |
|---|---|
| Optimizer | SGD, momentum=0.9 |
| Base learning rate | 0.002 |
| mlp.proj LR mult | 3.0 |
| mlp.fc LR mult | 0.5 |
| Intra-chunk cosine | Enabled |
| Epochs per chunk | 30 |
| Chunk size | 32,768 tokens |
| Stride | 64 |
| Frozen blocks | First 2 (of 11) |
| Trainable params | 19,911,748 / 24,634,452 |
| Eval time | ~3,690s (1×A100) |

## Quantization & Size

| Component | Bytes |
|---|---|
| Model (int6 + zstd) | 15,283,215 |
| Code (train_gpt.py) | 74,030 |
| **Total** | **15,357,245** |
| Limit | 16,000,000 |
| Headroom | 642,755 (4.0%) |

## Comparison to Prior Submissions

| Metric | PR #456 (1ep) | PR #461 (3ep) | PR #526 (30ep) | This (30ep+) | Δ vs #526 |
|---|---|---|---|---|---|
| **val_bpb** | 1.15321 | 1.14458 | 1.14252 | **1.13872** | **−0.00380** |
| Pre-TTT BPB | 1.1600 | 1.1611 | 1.1609 | 1.1574 (mean) | −0.0035 |
| TTT gain | −0.0068 | −0.0165 | −0.0184 | **−0.0182** | +0.0002 |
| Activation | ReLU² | ReLU² | ReLU² | **LeakyReLU(0.5)²** | new |
| Per-layer LR | No | No | No | **Yes** | new |
| Intra-cosine | No | No | No | **Yes** | new |

**Key insight**: The entire improvement comes from the better pre-TTT model (−0.0035 mean from LeakyReLU). Per-layer LR and intra-chunk cosine showed no measurable TTT improvement in this data — the TTT gain is −0.0182 vs −0.0184 in PR #526, a slight regression. These TTT modifications require further ablation to determine whether they help independently.

## Credits

This submission integrates work from many contributors to the parameter-golf competition:

- **LeakyReLU(0.5)²** — PR #518 (sofiabod): −0.004 BPB pre-TTT architecture improvement
- **Per-layer LR for TTT** — PR #481 (mrdavtan): differential learning rates for quantization-error recovery
- **Intra-chunk cosine LR** — PR #518 (sofiabod): cosine decay within each chunk's TTT epochs
- **30-epoch legal TTT** — Our prior work (PR #526): SGD + momentum + freeze-2
- **Score-first protocol** — Our prior work (PR #461): `torch.inference_mode()` during scoring
- **11L depth recurrence** — PR #455 / PR #442: shared BlockCores for weight-efficient depth
- **Partial RoPE, VE128, LN Scale** — PR #374 / PR #455: foundational architecture components
- **SmearGate, BigramHash, XSA** — Community contributions across multiple PRs
- **Muon optimizer** — PR #374 and descendants: Newton-Schulz orthogonal update for matrix params
- **SWA + Late QAT + int6/zstd** — Evolved across many PRs for quantization-aware training pipeline

## Reproducibility

```bash
# Environment: Python 3.10+, PyTorch 2.x with CUDA
# From the repo root:
RUN_ID=i39_leaky_perlr \
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
MLP_ACTIVATION=leaky_relu_sq \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=30 \
TTT_FREEZE_BLOCKS=2 TTT_BATCH_SEQS=32 TTT_MOMENTUM=0.9 \
TTT_PERLAYER_LR=1 TTT_PROJ_LR_MULT=3.0 TTT_FC_LR_MULT=0.5 \
TTT_INTRA_COSINE=1 \
SEED=7 \
torchrun --standalone --nproc_per_node=4 \
   records/track_non_record_16mb/2026-03-23_11L_LeakyReLU_PerLayerLR_LegalTTT/train_gpt.py
```
