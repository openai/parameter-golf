# Core Techniques (The "Core Five")

These five techniques are near-universal across all competitive submissions and collectively worth ~0.05-0.07 BPB.

## 1. Int6 Quantization

- 6-bit integer quantization (64 levels, range [-32, 31]) with per-row scale factors
- Frees ~25% more artifact space than int8, reinvested in bigger models
- Sensitive layers (tied embedding) kept in FP16 to avoid compounding errors
- **Origin:** @nanlliu in PR #39

## 2. MLP 3x Expansion

- Increase MLP hidden dim from 2x (1024) to 3x (1536) for 512-dim model
- More expressive nonlinear capacity; enabled by int6 space savings
- **Origin:** @jfprincz in PR #70, independently by @saml212 in #61

## 3. Sliding Window Evaluation

- Overlapping windows (stride=64, window=2048) ensure every token has rich context
- Pure eval-time technique, no model change
- Worth ~0.034 BPB (from #77 ablation)
- **Stride debate:** stride=256 gives marginally better BPB at 4x less eval time
- Doc isolation hurts at stride=64 (tokens lose context at boundaries)
- **Origin:** @mattqlf in PR #50

## 4. FP16 Tied Embedding

- Keep the embedding matrix (used for both input and output) in FP16 instead of quantizing
- Errors compound in both directions (input + output); highest-value precision decision
- ~1MB cost, disproportionate impact
- **Origin:** @chonchiog in PR #42

## 5. Zstd-22 Compression

- Zstandard level 22 compresses int6 weights much tighter than zlib
- Frees ~1-2MB for more parameters — effectively free lunch
- Compression is one-time after training; decompression is fast
- Every int6 submission uses zstd-22

## Consensus Optimizer Settings

- **Muon** optimizer with momentum 0.99 (warmup from 0.92 over 1500 steps)
- LR: matrix=0.02, scalar=0.02, embed=0.03
- Warmdown: 3000 iters
- Grad clip: 0.3
- **Exception:** #76 uses higher LRs (0.03) and lower momentum (0.97)
- **Exception:** #236 uses 524K batch (vs 786K) — 22% more gradient updates, worth 0.017 BPB

## Top Stack Additions

- SmearGate + BigramHash + OrthoInit (requires OrthoInit to work)
- 11 layers + WD 0.04
- Weight averaging (SWA or EMA depending on architecture)
- Frontier (#315): + Partial RoPE, LN Scale, Late QAT, XSA on last 4 layers, EMA (0.997)
