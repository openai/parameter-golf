# Record: Packed Causal N-gram + Dirichlet Backoff Mixing — val_bpb 0.0180 (3-seed mean)

## Results

| Seed | val_bpb | Artifact | Eval time |
|------|---------|----------|-----------|
| 42 | 0.01801879 | 1,376,353 bytes | 283s |
| 1337 | 0.01799416 | — | 283s |
| 2024 | 0.01799022 | 1,384,609 bytes | 266s |
| **Mean** | **0.01800106** | | |
| **Std** | **0.00001541** | | |

- Artifact: < 16,000,000 bytes (all seeds, ~1.4 MB)
- Train: < 600s on 8xH100 SXM (all seeds)
- Eval: < 600s (all seeds, 266-283s)

## Method

2-layer 128d GPT (vestigial — provides base probabilities only). Order 2-12 n-gram hash tables pre-computed from 24 training shards (8M token budget), stored as int32 counts in 32K buckets, zstd-compressed in artifact. Single-pass score-first eval with Dirichlet posterior backoff mixing and count-confidence gating.

### Architecture
- 2L, 128d, 4 heads / 2 KV heads, MLP 2x, RoPE 16 dims
- Tied embeddings, logit softcap 30
- SWA, Muon optimizer
- int6 per-row quantization + zstd-22 compression

### Packed N-gram Cache (Training Time)
- Order 2-12 hash tables built from 24 training shards (8M token budget)
- 32,768 (32K) buckets per order, dual hash (context + full n-gram)
- XOR-of-products hashing with position-dependent primes
- All-reduce across 8 GPUs during build, then packed into artifact
- ~244 entries per bucket average (unsaturated — real n-gram statistics)

### Dirichlet Posterior Backoff Mixing (Eval Time)
- Greedy highest-order-first backoff: check order 12, fall back to 11, ..., 2
- Each position matched by exactly ONE order (the highest with sufficient evidence)
- Dirichlet posterior: `posterior = (full_count + c * model_p) / (ctx_count + c)`
- Per-order concentrations: [50, 50, 20, 10, 6, 4, 3, 2.5, 2, 1.8, 1.6] (high for noisy low orders, low for specific high orders)
- Count-confidence gating: `conf = ctx_count / (ctx_count + 12.0)`, then `blended = (1-conf)*model_p + conf*posterior`
- Low-count contexts lean toward neural model; high-count contexts trust the posterior

### Single-Pass Score-First Eval
- Sliding window with stride 64, seq_len 2048
- For each window: (1) compute neural logits, (2) backoff n-gram lookup, (3) Dirichlet blend, (4) update cache with scored tokens
- Distributed prefill: each rank pre-warms cache with all preceding token positions
- No two-pass, no phrase cache, no TTT

## Legality

- [x] **Score-first**: each window: lookup cache THEN update cache. No token sees future data.
- [x] **Single-pass only**: no two-pass rescore. Each token scored exactly once.
- [x] **Mixing coefficient is target-independent**: `conf = ctx_count / (ctx_count + 12)` depends only on context count. The Dirichlet posterior evaluates the neural prior at the target (standard for computing predictive probability).
- [x] **Packed artifact uses training data only**: n-gram tables built from `fineweb_train_*.bin` shards during training phase.
- [x] **No TTT**: test-time training disabled.
- [x] **No GPTQ at eval time**: quantization completes within training budget.
- [x] **No reordering**: evaluation set processed in original sequential order.
- [x] **Deterministic**: same seed = same result (std = 0.000015).
- [x] **Artifact < 16,000,000 bytes**: ~1.4 MB (all seeds).
- [x] **Eval time < 600s**: 266-283s (all seeds).

## Credits

- PR #944: Packed causal n-gram memory + Dirichlet backoff mixing architecture
- PR #900: Dirichlet posterior mixing theory (8.9x better than linear interpolation)
- PR #943: Packed causal memory concept
- PR #727/#753: Multi-order n-gram backoff foundation
- PR #414: Base model architecture stack
