# Coprime-Stride Loader + Full GPTQ + Score-First TTT

**val_bpb: 1.08008** (3-seed mean, std 0.0009) | **~15.99 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Steps | Pre-quant BPB | Quantized BPB | Sliding BPB | TTT BPB | Artifact |
|------|-------|---------------|---------------|-------------|---------|----------|
| 1337 | 4565 | 1.08596 | 1.09737 | 1.08058 | **1.07907** | 15,992,892 |
| 42   | 4570 | 1.08742 | 1.09877 | 1.08217 | **1.08075** | 15,996,411 |
| 2025 | 4566 | 1.08722 | 1.09874 | 1.08196 | **1.08043** | 15,993,485 |
| **Mean** | **4567** | **1.08686** | **1.09829** | **1.08157** | **1.08008** | **15,994,263** |

## Key Innovations

### 1. Coprime-Stride Multi-Shard Loader
Replaces the standard `ShuffledSequenceLoader` with a coprime-stride data loader (PR #726 style). Within each shard, sequences are accessed with a stride coprime to the block count, guaranteeing every block is visited exactly once per epoch without cyclic patterns. Adaptive shard selection uses progress-based weighting (alpha decays from 0.9 to 0.5) with interleaved bucket draining for maximum diversity per batch.

**Effect:** +36 extra training steps (4565 vs 4529 baseline), better pre-quant BPB (1.0860 vs 1.0866).

### 2. Full Hessian GPTQ with Cholesky Fallback
Standard GPTQ with Cholesky error compensation + actorder (column sorting by Hessian diagonal). SD-based clipping at 12.85σ for int6 matrices, 20σ for int8 embeddings. Added Cholesky fallback: if `torch.linalg.cholesky` fails on an ill-conditioned Hessian, falls back to simple per-row quantization instead of crashing.

### 3. LZMA Code Compression
Full Python source (53KB) is LZMA-compressed + base85-encoded into a 2-line self-extracting .py file (18KB). Saves ~35KB in artifact size, keeping total under 16MB. Same technique as the current SOTA record.

### 4. Score-First TTT (Legal)
Score-first per-chunk test-time training following the PR #461/#549 framework:
- Score each 32K-token chunk under `torch.no_grad()` first
- Then train on that chunk with SGD (momentum=0.9, LR=0.005, 3 epochs)
- Adapted model only scores future chunks — never rescores tokens it trained on

## Architecture

- SP8192 BPE tokenizer (8192 tokens)
- 11 physical layers, 17 virtual (depth recurrence: layers 3-5 looped 3×)
- dim=512, 8 heads, 4 KV heads (GQA), MLP 4× with LeakyReLU(0.5)²
- XSA on all 11 layers, parallel residuals from layer 7+
- U-Net skip connections with learnable gates
- Tied embeddings, logit softcap=30

## Training

- Muon optimizer (5-step Newton-Schulz) + AdamW for embeddings/scalars
- EMA (decay 0.9965)
- 72% warmdown, 20-step warmup + 20-step loop warmup
- Gradient clipping at 0.3
- Brotli-11 compression + byte shuffling

## Compliance

### Condition 1 (Strict Causal Dependence)
Causal attention via `flash_attn_func(causal=True)`. TTT only incorporates tokens from already-scored chunks.

### Condition 2 (Full Normalized Distribution)
Standard `F.cross_entropy` over full vocab_size logits. No top-k masking.

### Condition 3 (Score-Before-Update)
Each chunk scored under `torch.no_grad()` before any training on that chunk. Model weights at scoring time reflect only prior chunks.

### Condition 4 (Single Left-to-Right Pass)
Single `for ci in range(num_chunks)` loop. Each token scored exactly once. No rescoring or min-over-runs.

## Credits
- SOTA base: PR #1394 by @clarkkev (Full Hessian GPTQ + SDClip)
- Depth recurrence: PR #1445 by @dexhunter
- Score-first TTT: PR #549 by @abaybektursun, PR #461 by @Christopher-Lee-McClendon
- Coprime-stride loader: PR #726 style
- XSA: PR #634
- Parallel residuals: PR #1412 by @Robby955
- LZMA code compression: PR #1394 technique
