# Knowledge Required — Parameter Golf

This document maps the knowledge domains you need to compete effectively.

---

## 1. Transformer Architecture Fundamentals

**Why**: Every technique in the leaderboard is a modification to the standard transformer.

| Concept | Where it shows up in the challenge |
|---------|------------------------------------|
| Attention (MHA → GQA) | Baseline uses 8Q / 4KV heads |
| MLP (FFN) design | MLP_MULT, activation choice (GELU → LeakyReLU²) |
| RoPE positional encoding | Partial RoPE, YaRN base tuning |
| RMSNorm / LayerNorm | LN scale tricks (1/√(layer+1)) |
| Residual connections | U-Net skip connections, residual mixing |
| Tied embeddings | Saves parameters; tied LR is tuned separately |
| KV caching | Matters for evaluation / TTT |

**Resources**: Attention Is All You Need, Karpathy's nanoGPT, LLaMA paper.

---

## 2. Quantization

**Why**: The 16MB budget is the core constraint. Quantization is how you squeeze a 50–100M param model into it.

| Technique | What to know |
|-----------|-------------|
| INT8 per-row quantization | Default in baseline; clip percentile tuning |
| INT6 STE QAT | Quantization-aware training with straight-through estimator |
| INT5 / ternary / 1-bit | Extreme quantization schemes; requires careful activation tuning |
| GPTQ (diagonal vs full Hessian) | Post-training quantization with second-order info |
| Zlib / Zstandard / LZMA | Compression after quantization; LZMA preset=9 saves ~0.5% |
| Block quantization | Group quant for better accuracy at same bit-width |

**Key insight**: Quantization + compression = artifact size. Lower artifact = more room for parameters.

---

## 3. Parameter Efficiency

**Why**: More effective parameters within the 16MB budget = lower BPB.

| Technique | Effect |
|-----------|--------|
| BigramHash embedding | Adds bigram co-occurrence info at tiny param cost |
| SmearGate | Position-mixing gate that propagates context info |
| U-Net skip connections | Encoder-decoder shortcuts across layers |
| Cross-Sequence Attention (XSA) | Cross-position mixing at zero parameter cost |
| Low-rank projections | Reduce weight matrix size while preserving rank |
| Parameter tying | Share weights between layers (depth recurrence) |

---

## 4. Optimization (Muon + Adam)

**Why**: The optimizer directly controls how well the model uses its parameters.

| Concept | Where it matters |
|---------|-----------------|
| Muon optimizer | Used for matrix params; Newton-Schulz orthogonalization |
| Parallel Muon | Cross-GPU gradient sharing trick (PR #399) |
| Muon momentum warmup | 0.85 → 0.95; important for early stability |
| Adam for scalars | LR split: embed_lr=0.6, matrix_lr=0.04 |
| LR schedule | Cosine warmdown; top runs use WARMDOWN_ITERS=4000 |
| Weight averaging | EMA(0.997) + SWA every 50 steps |
| Gradient clipping | Usually disabled (GRAD_CLIP_NORM=0) |

---

## 5. Evaluation & Scoring

**Why**: The metric is BPB (bits-per-byte), not cross-entropy loss. These are not the same.

| Concept | Detail |
|---------|--------|
| BPB formula | `cross_entropy_bits × tokens_per_byte` |
| Tokenizer-agnostic scoring | BPB normalizes for vocabulary size differences |
| Sliding window evaluation | Shift eval window to capture longer contexts; −0.02 BPB gain |
| Statistical significance | Welch's t-test across 3 seeds, p < 0.01 required |
| Artifact size check | `code_bytes + lzma(model_bytes) ≤ 16,000,000` |

---

## 6. Distributed Training (PyTorch DDP / torchrun)

**Why**: The 10-minute limit on 8×H100 requires efficient multi-GPU training.

| Concept | Detail |
|---------|--------|
| `torchrun --nproc_per_node=8` | Launches 8 processes, one per GPU |
| `dist.all_reduce` | Aggregates gradients across ranks |
| Gradient accumulation | `TRAIN_BATCH_TOKENS / (world_size × seq_len)` |
| Flash Attention 3 | Hopper warp-specialized kernels; required for top runs |
| BF16 training | Default dtype for activations; weights in FP32 |

---

## 7. Test-Time Training (TTT)

**Why**: One technique (PR #549) used TTT for −0.0025 BPB.

| Concept | Detail |
|---------|--------|
| Score-first TTT | Sort eval tokens by model confidence; adapt on easy ones first |
| LoRA TTT | Fine-tune low-rank adapters at test time |
| Legal TTT | Can only train on tokens already evaluated (not future val data) |

---

## 8. Compression Algorithms

| Algorithm | Typical ratio on int8 model | Notes |
|-----------|---------------------------|-------|
| zlib (default) | ~1.15× | Fast, decent |
| zstandard (level 22) | ~1.20× | Better than zlib for model weights |
| LZMA (preset 9) | ~1.25× | Best compression, slowest |

Switch compression in `export_artifact()` in your `train_gpt.py` fork.

---

## 9. Git / PR Workflow for the Challenge

- Submissions are PRs that only add a new folder to `/records`
- PR creation time determines chronological priority for the leaderboard
- Reproducibility is checked; keep your `train_gpt.py` self-contained
- Add `requirements.txt` for any non-standard packages

---

## Learning Path (Recommended Order)

1. Read nanoGPT code → understand baseline
2. Read Muon paper / blog post → understand optimizer
3. Read top 3 record READMEs → understand the stack
4. Run baseline locally → get hands-on
5. Pick one technique, read its originating PR → implement and measure
6. Stack techniques incrementally; measure each delta
