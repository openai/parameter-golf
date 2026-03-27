# Order-13 Full-Rescore N-gram + 11L Int6 GPTQ

**val_bpb: 0.09378 (3-seed mean, std 0.00003)**

## Results

| Seed | val_bpb | eval_time | artifact_bytes |
|------|---------|-----------|----------------|
| 1337 | 0.09380 | 247s | 15,774,084 |
| 42 | 0.09379 | 246s | ~15.7MB |
| 2026 | 0.09374 | 576s | ~15.8MB |
| **Mean** | **0.09378** | | |
| **Std** | **0.00003** | | |

## Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3.5x expansion with LeakyReLU(0.5)^2
- Gated attention + value residual + XSA-all
- Partial RoPE (64 dims), value embeddings on layers 8-10
- BigramHash (1024 vocab, 256 dim)
- Tied embeddings, logit softcap=20

## Training

- 8xH100 SXM, 600s wallclock
- EMA(0.997), warmdown=3500 steps
- Muon optimizer (matrix_lr=0.05) + AdamW for embeddings/scalars
- Late QAT at scale < 0.15

## Quantization

- Int6 GPTQ with **descending actorder** + dead column handling
- lzma(8) compression
- 3% magnitude pruning
- GPTQ retry loop (4 attempts with +1% prune per retry)

## N-gram Eval Cache (the key innovation)

Two-pass order-13 backward-looking n-gram cache:

**Pass 1 (score-first, legal):**
- Process validation tokens in 1M-token chunks
- For each chunk: model forward pass → score tokens → update n-gram cache
- Cache only contains already-scored tokens (backward-looking)
- Captures per-token model probabilities and entropy for Pass 2

**Pass 2 (full-rescore):**
- No additional model forward passes
- Re-score all tokens using the COMPLETE n-gram cache
- Entropy-adaptive mixing: α = sigmoid(scale * (entropy - center)) with order-shifted centers
- Per-order multipliers: 0.3x for bigram/trigram, 2x for 5-gram+
- α_min=0.05, α_max=0.60, entropy_center=3.0, entropy_scale=2.0

**Implementation:**
- Pure NumPy with vectorized batch operations
- XOR-of-products hashing with 14 primes
- 4M buckets (power-of-2 masking)
- np.bincount for O(n) bulk cache updates
