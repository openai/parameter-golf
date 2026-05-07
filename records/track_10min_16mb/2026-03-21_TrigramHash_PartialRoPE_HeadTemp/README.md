# TrigramHash + PartialRoPE + Per-Head Temperature + Stride32

**val_bpb: 1.1450** (mean of 2 seeds: 1.1449, 1.1451)

## Approach

Built on the 10L Int5-MLP + BigramHash(10240) + SWA foundation from PR #162 and the 10L Int5-MLP submission by @thwu1. We add five independent improvements, each contributing a small but measurable gain.

## Novel Contributions

### 1. TrigramHashEmbedding (novel)
Extends the BigramHash idea to consecutive token triplets. A hash table of 8192 buckets (dim=64, projected to model_dim=512) maps `(tok[i-2]*961 + tok[i-1]*31 + tok[i]) % 8191 + 1` to learned embeddings. This captures 3-token patterns ("in the morning", "at the end") as atomic units, complementing BigramHash which only sees pairs. Output is gated by a learned scalar initialized at zero for stable training.

### 2. Partial RoPE (50% of head dimensions)
Rotary position embeddings are applied to only the first 50% of each attention head's dimensions. The remaining dimensions are position-free, enabling the model to find similar tokens regardless of their position in the sequence. This improves length generalization and helps the sliding window evaluation extrapolate better.

### 3. Per-Head Temperature Scaling
Each attention head learns its own temperature parameter (initialized to 1.0), multiplied after the existing q_gain. This allows some heads to develop sharp, focused attention patterns while others maintain broad, contextual attention — increasing the diversity of attention behaviors without adding significant parameters.

### 4. Eval Stride 32 (reduced from 64)
Sliding window evaluation stride reduced from 64 to 32 tokens, providing finer-grained overlapping context for each evaluation window. This gives every token more preceding context on average, improving prediction quality at the cost of ~2x eval time (still well within the 10-minute eval budget).

### 5. LoRA TTT Infrastructure
Added complete LoRA-based test-time training framework (LoRAAdapter, attach/remove helpers, eval_val_with_ttt). Attaches rank-4 LoRA adapters to all projection layers, trains on the first half of each eval chunk, then scores the full chunk. Currently disabled by default (TTT_ENABLED=0) pending further tuning but infrastructure is ready.

## Results

| Seed | val_bpb | Artifact Size |
|------|---------|---------------|
| 42   | 1.1449  | ~15.9 MB      |
| 1337 | 1.1451  | ~15.9 MB      |
| Mean | 1.1450  | -             |
| Std  | 0.0001  | -             |

## Architecture

- 10 transformer layers, 512 model dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128) + TrigramHash(8192, dim=64)
- Partial RoPE (50% dims) + Per-head temperature scaling
- U-Net skip connections, tied embeddings
- Int5 MLP / Int6 attention q
