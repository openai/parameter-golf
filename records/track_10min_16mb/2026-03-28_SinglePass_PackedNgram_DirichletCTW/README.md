# Record: Single-Pass Packed N-gram + Hierarchical Dirichlet CTW — val_bpb 0.1130 (3-seed mean)

## Results

| Seed | val_bpb | Artifact | Eval time |
|------|---------|----------|-----------|
| 42 | 0.11300057 | 5,757,313 bytes | 331s |
| 1337 | 0.11300056 | 5,759,723 bytes | 354s |
| 2024 | 0.11300055 | 5,757,266 bytes | 332s |
| **Mean** | **0.11300056** | | |
| **Std** | **0.00000001** | | |

- Artifact: < 16,000,000 bytes (all seeds)
- Train: < 600s on 8xH100 SXM (all seeds)
- Eval: < 600s (all seeds)

## Method

2-layer 128d GPT (vestigial — provides base probabilities only). Order 2-13 n-gram hash tables pre-computed from 80 training shards (10B tokens), stored as uint16 counts in 128K buckets, zstd-compressed in artifact. Single-pass score-first eval with hierarchical Dirichlet CTW mixing (per-order concentrations). No two-pass rescore. Cache is deterministic — BPB variance across seeds is < 1e-7.

### Architecture
- 2L, 128d, 4 heads / 2 KV heads, MLP 2x, RoPE 16 dims
- Tied embeddings, logit softcap 30
- SWA, Muon optimizer
- int6 per-row quantization + zstd-22 compression

### Packed N-gram Artifact
- Order 2-13 hash tables built from ALL 80 training shards during training phase
- 131,072 (128K) buckets per order, dual hash (context + full n-gram)
- uint16 counts, ratio-preserving scaling, zstd-compressed
- All-reduce across 8 GPUs during build, then packed into artifact
- At eval: cache starts instantly warm with billions of training observations

### Hierarchical Dirichlet CTW Mixing
- Per-order concentrations: [50, 50, 20, 10, 6, 4, 3, 2.5, 2, 1.8, 1.6, 1.4] (high for noisy low orders, low for specific high orders)
- Each order's Dirichlet posterior becomes the next order's prior
- Formula: `blended[i] = (c * prev_p + full_count) / (c + ctx_count)`
- Based on Context Tree Weighting (Willems et al. 1995) and Dirichlet-Multinomial posterior predictive (Teh 2006)

### Single-Pass Score-First Eval
- Sliding window with stride 128, seq_len 2048
- For each window: (1) lookup prewarmed cache, (2) compute Dirichlet-blended loss, (3) update cache with scored tokens
- Distributed prefill: each rank pre-warms with all preceding token positions
- No second pass — every token scored exactly once, no self-inclusion

## Key Innovation

The packed n-gram artifact eliminates the cold-start problem that plagues online-only n-gram caches. By pre-computing hash tables from 10B training tokens and storing them in the 16MB artifact, the cache starts with high-quality statistics from the first eval token. Combined with hierarchical Dirichlet CTW mixing (which is provably optimal for backoff smoothing), this produces a 0.1130 BPB result using single-pass only — no two-pass rescore, no self-inclusion risk.

## Legality

- [x] **Score-first**: each window: lookup cache THEN update cache. No token ever sees its own contribution.
- [x] **Single-pass only**: no two-pass rescore, no self-inclusion. Each token scored exactly once.
- [x] **Packed artifact uses training data only**: n-gram tables built from training shards during training phase. No validation data in artifact.
- [x] **Dirichlet mixing depends on counts only**: no dependence on target token identity for mixing weights.
- [x] **No TTT**: test-time training disabled (TTT_EPOCHS=0).
- [x] **No GPTQ at eval time**: quantization completes within training budget.
- [x] **No reordering**: evaluation set processed in original sequential order.
- [x] **Deterministic**: same seed = same result (std = 0.00000001 across seeds).
- [x] **Artifact < 16,000,000 bytes**: 5.76 MB (all seeds).
- [x] **Eval time < 600s**: 331-354s (all seeds).

## Credits

- PR #900: Dirichlet posterior mixing theory and ablation proving 8.9x superiority over linear interpolation
- PR #943: Packed causal n-gram memory concept and per-order concentration formula
- PR #880: Variable-length phrase cache architecture (not used here but informed design)
- PR #727/#753: Multi-order n-gram backoff with entropy-adaptive alpha (foundation)
- PR #414: Base model architecture stack
- Willems et al. (1995): Context Tree Weighting
- Teh (2006): Hierarchical Dirichlet processes for language modeling
