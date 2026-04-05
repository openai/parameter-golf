# 10-Layer Gated DeltaNet (PureGDN) with Legal Score-First TTT

**val_bpb = 1.003028** (3-seed mean) | Pre-TTT: 1.007803 | TTT gain: **−0.004775** | Artifact: 15.17 MB decimal (14.47 MiB)

> Non-record unlimited-compute submission (trained 7k steps on 2×A100-40GB, eval ~2100s on 1×A100).

---

## Headline Result

This submission replaces softmax attention with **Gated DeltaNet (GDN)** — a linear-attention mechanism from the FLA (Flash Linear Attention) library that uses delta-rule-based gating for $O(n)$ sequence processing. The overall architecture still follows the standard transformer block pattern (pre-norm → token mixing → MLP → residual), but uses GDN's delta-rule linear attention in place of softmax attention. With 10 GDN layers, BigramHash embeddings with trigram extension, and score-first TTT, the model achieves **1.003028 BPB**.

These results suggest that a GDN-based linear-attention backbone can be competitive on this benchmark when paired with the training, quantization, and evaluation recipe used here.

---

## Novel & Creative Contributions

### 1. Gated DeltaNet Backbone

Most accepted leaderboard entries in the repo README appear to use standard softmax attention. This submission instead uses **Gated DeltaNet** (Yang et al., 2024), a linear-attention variant from FLA v0.4.2. The GDN paper describes this family as "Linear Transformers" — the overall block structure (pre-norm, token mixing, MLP, residual connections) is the same as a standard transformer, but softmax attention is replaced with delta-rule-based gating that selectively writes to and erases from a fixed-size state, enabling $O(n)$ sequence processing with sub-quadratic memory.

- **dim=512**, 1 head, expand_k=1, expand_v=2
- No softmax attention; uses GDN's delta-rule linear attention for token mixing
- FLA's Triton kernels provide efficient chunk-wise parallel training

### 2. Sub-1.01 BPB without Softmax Attention

The model achieves **1.003028 BPB** (3-seed mean), showing that replacing softmax attention with GDN's linear attention can be competitive on this benchmark. Rather than claiming a general result about attention mechanisms, we view this as evidence that softmax attention is not the only viable token-mixing mechanism for strong performance in this particular setting.

### 3. BigramHash with Trigram Extension

The standard BigramHash(vocab, dim) from PR #65 hashes consecutive token pairs for cheap bigram context. We extend this with an additive **trigram hash channel** that captures 3-gram patterns, giving the model cheap trigram context without additional learned parameters.

- BigramHash: vocab=3072, dim=112
- Trigram hash: additive channel on top of bigram features

### 4. Legal TTT on Linear Attention

In this GDN-based submission, **score-first TTT** (SGD + momentum, 3 epochs, freeze first 2 blocks) produced a consistent improvement across all three seeds. The mean TTT gain of **−0.004775 BPB** suggests that this protocol can also be effective in a linear-attention setting.

---

## 3-Seed Results

| Seed | SLURM Job | Baseline BPB | Legal TTT BPB | TTT Δ | Artifact (bytes) |
|------|-----------|-------------|---------------|-------|-------------------|
| 42   | 55486643  | 1.00871911  | 1.00381390    | −0.004905 | 15,170,538 |
| 1337 | 55511207  | 1.00726155  | 1.00245252    | −0.004809 | 15,143,381 |
| 2025 | 55511287  | 1.00742810  | 1.00281683    | −0.004611 | 15,172,519 |
| **Mean** | | **1.007803 ± 0.000798** | **1.003028 ± 0.000705** | **−0.004775** | |

### Context vs. Current 10-Minute Record

- For context, this **non-record unlimited-compute** result is **0.111707 BPB** lower than the current 10-minute record-track entry (1.11473509 BPB, PR #1019).
- This is not a like-for-like record comparison because the present submission is in the non-record track and is not constrained by the 10-minute training limit.

---

## Architecture Summary

| Component | Configuration |
|---|---|
| Layers | 10 (all GDN, no transformer layers) |
| Layer Type | Gated DeltaNet (FLA v0.4.2) |
| Embedding dim | 512 |
| GDN heads | 1, expand_k=1, expand_v=2 |
| MLP | SwiGLU 3× expansion (1536), CastedLinear |
| Vocab | 1024 (SentencePiece BPE) |
| BigramHash | 3072 features, 112 dim, trigram extension |
| Normalization | RMSNorm (pre-norm) |
| Tied embeddings | Yes |
| Parameters | 29,926,689 total |

**What this model does NOT have:** No RoPE, no XSA, no U-Net skips, no value embeddings, no depth recurrence. Uses GDN linear attention in place of softmax attention, but retains the standard transformer block pattern (pre-norm + token mixing + MLP + residual).

## Training Details

| Setting | Value |
|---|---|
| Hardware | 2×A100-PCIE-40GB (NVIDIA, NCSA Delta HPC) |
| Steps | 7,000 |
| Training wallclock | ~4.8h per seed |
| Optimizer | Muon (matrix params) + Adam (embeddings/scalars) |
| EMA | decay=0.95, started at step 3500 |
| SWA | 12 checkpoints from step 6450 |
| Late QAT | Enabled at step 6601 |
| Quantization | Int6 + GPTQ + zstd-22 |
| Sequence length | 1024 |
| Batch size | 64 sequences = 65,536 tokens |

## TTT Protocol (Legal Score-First)

```
for each 32K-token chunk:
    1. model.eval() + torch.inference_mode()
       → Forward pass on chunk, accumulate NLL    ← SCORE (graded)
    2. model.train()
       → SGD(lr=0.002, momentum=0.9), 3 epochs   ← TRAIN (adaptation)
    3. Advance to next chunk with updated weights
```

In this implementation, every target token is intended to be scored before any gradient update that could benefit from it. The `torch.inference_mode()` context manager prevents autograd-based gradient accumulation during the scoring pass.

| TTT Setting | Value |
|---|---|
| Optimizer | SGD, momentum=0.9 |
| Learning rate | 0.002 |
| Epochs per chunk | 3 |
| Chunk size | 32,768 tokens |
| Stride | 64 |
| Frozen blocks | First 2 (of 10) |
| Eval time | ~2100s (1×A100) |

## Quantization & Size

| Component | Bytes |
|---|---|
| Model (int6 + GPTQ + zstd-22) | 15,170,538 (seed 42; max across seeds: 15,172,519 = 15.17 MB decimal / 14.48 MiB) |
| Code (train_gdn_7k.py + architectures.py + configs.py + eval_ttt.py + requirements.txt) | 104,167 |
| **Total** | **15,274,705** |
| Limit | 16,000,000 |
| Headroom | 725,295 (4.5%) |

## Eval Legality

We reviewed the evaluation protocol against the contest's score-before-train rule and believe it satisfies that constraint:

- **Score before train:** Each validation segment is scored before any weight update that could use those targets. The `torch.inference_mode()` context during scoring prevents autograd-based gradient accumulation in that phase.
- **Fresh model state:** A fresh model state is loaded per evaluation (no training state carry-over between seeds).
- **Deterministic windowing:** Window assignment is deterministic based on `scored_start`, ensuring no gaps or overlaps in token scoring.
- **Stateless normalization:** RMSNorm is stateless (no running statistics unlike BatchNorm), so layer normalization cannot leak future information.
- **Last-chunk guard:** The last chunk is never trained on (conservative guard against edge-case leakage).
- **Minor notes:**
  - SGD momentum carries across chunks — this is an optimization state update, not an information leak (momentum contains only gradient history from already-scored tokens).
  - Cosine LR schedule is deterministic and independent of validation data.

## Comparison to Prior Submissions (Context Only)

| Metric | 10-min SOTA (PR #1019) | Non-record best (Binary UNet, single seed) | This Submission |
|---|---|---|---|
| val_bpb | 1.11473509 | 1.1239 | **1.003028** |
| Architecture | 11L Softmax Transformer | 15L Softmax Transformer (UNet) | **10L Linear-Attention Transformer (GDN)** |
| Token Mixing | Softmax Attention | Softmax Attention | **Delta-Rule Linear Attention (GDN)** |
| TTT | None | — | **Legal TTT** |

These comparisons are included only as context. They are not claims of a like-for-like record result across tracks or compute budgets.

## Reproducibility

```bash
# Environment: Python 3.12, PyTorch 2.x with CUDA, FLA v0.4.2
pip install -r requirements.txt  # installs flash-linear-attention + zstandard

# Training (2×A100, ~4.8h per seed):
SEED=42 torchrun --standalone --nproc_per_node=2 train_gdn_7k.py
# Or via SLURM: see supplementary/slurm_train_2xA100.sh

# Evaluation with TTT (1×A100, ~35min):
ARTIFACT_PATH=final_model.int6.ptz ARCH_MODE=A TTT_ENABLED=1 \
  TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=2 \
  python eval_ttt.py
# Or via SLURM: see supplementary/slurm_eval_legal_ttt.sh
```

## Code Structure

| File | Lines | Description |
|---|---|---|
| `train_gdn_7k.py` | 1,162 | Main training script (data loading, optimizer, training loop, GPTQ, SWA) |
| `architectures.py` | 661 | Model definition (HybridGDN, RecurrentBlock, MLP, BigramHash, CastedLinear) |
| `configs.py` | 202 | Architecture configurations (model A = PureGDN used here) |
| `eval_ttt.py` | 467 | Evaluation with sliding-window + legal score-first TTT |
| `requirements.txt` | — | Extra pip deps beyond the repo-root requirements.txt |

## Supplementary Files

```
supplementary/
├── eval_seed42_legal_ttt.log       # Eval log for seed 42
├── eval_seed1337_legal_ttt.log     # Eval log for seed 1337
├── eval_seed2025_legal_ttt.log     # Eval log for seed 2025
├── slurm_eval_legal_ttt.sh         # SLURM script for eval jobs
├── slurm_train_2xA100.sh           # SLURM script for training
└── seed_runs/
    ├── train_s42.log               # Training log for seed 42
    ├── train_s1337.log             # Training log for seed 1337
    └── train_s2025.log             # Training log for seed 2025
```

## Credits

This submission builds on work from many contributors to the parameter-golf competition:

- **Gated DeltaNet (GDN)** — Yang et al., "Gated Delta Networks" (2024); FLA library by Songlin Yang et al. (fla-org/flash-linear-attention v0.4.2)
- **Muon optimizer** — Baseline (`modded-nanogpt`); Newton-Schulz orthogonal preconditioning
- **BigramHash embeddings** — PR #65 (aquariouseworkman): hash consecutive token pairs for cheap bigram context
- **SmearGate** — PR #65 (aquariouseworkman): per-dim sigmoid gate blending adjacent token embeddings (used in BigramHash embedding)
- **Legal TTT framework** — PR #77 (samacqua): first legal score-first TTT (LoRA); full-model SGD variant in our earlier PR #456 (Christopher-Lee-McClendon)
- **TTT recipe (SGD + momentum + freeze)** — PR #461 (our own): SGD momentum 0.9, 3 epochs, freeze first 2 blocks
- **Mixed int5/int6 quantization** — PR #76 (unixmadtoonslab / Will DePue): int6 weight quantization
- **SWA (Stochastic Weight Averaging)** — PR #69 (TevBenji): checkpoint averaging during warmdown
- **Late QAT** — PR #315 (jfprincz), working implementation in PR #374 (unnir): STE fake-quantization in final training phase
- **GPTQ calibration** — PR #1019 (abaybektursun): autoregressive self-generated calibration
- **Sliding window evaluation** — PR #50 (mattqlf / Matthew Li): stride-64 overlapping windows
- **ReLU² activation** — Baseline (`modded-nanogpt`) (heritage; this submission uses SwiGLU)
- **EMA (Exponential Moving Average)** — Standard technique; decay=0.95 starting at step 3500

Built on the [parameter-golf](https://github.com/openai/parameter-golf) starter code by Beren Millidge & Keller Jordan.
