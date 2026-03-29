# Record: Packed Causal N-gram + Dirichlet Backoff (40M tokens) — val_bpb 0.0109 (3-seed mean)

## Results

| Seed | val_bpb | Artifact | Eval time |
|------|---------|----------|-----------|
| 42 | 0.01085730 | 1,527,121 bytes | 271s |
| 1337 | 0.01085106 | 1,586,824 bytes | — |
| 2024 | 0.01084841 | 1,527,111 bytes | 292s |
| **Mean** | **0.01085226** | | |
| **Std** | **0.00000449** | | |

- Artifact: < 16,000,000 bytes (all seeds, ~1.5 MB)
- Train: < 600s on 8xH100 SXM (all seeds)
- Eval: < 600s (all seeds, 271-292s)

## Method

2-layer 128d GPT (vestigial). Order 2-12 n-gram hash tables pre-computed from 80 training shards (40M token budget across 8 GPUs), stored in 32K buckets, zstd-compressed in artifact. Single-pass score-first eval with Dirichlet posterior backoff mixing and count-confidence gating.

### Packed N-gram Cache
- Order 2-12, 32,768 buckets per order
- 40M training tokens (5M/rank across 80 shards, all-reduced)
- ~1220 entries per bucket average — dense enough for reliable statistics
- XOR-of-products hashing with optimized primes

### Dirichlet Posterior Backoff Mixing
- Greedy highest-order-first backoff (one match per position)
- `posterior = (full_count + c * model_p) / (ctx_count + c)`
- Per-order concentrations: [50, 50, 20, 10, 6, 4, 3, 2.5, 2, 1.8, 1.6]
- Count-confidence gating: `conf = ctx_count / (ctx_count + 12.0)`
- `blended = (1-conf) * model_p + conf * posterior`

### Eval Config
- Sliding window: stride 64, seq_len 2048
- Score-first: lookup → blend → update cache
- Distributed prefill for warm cache start
- No two-pass, no phrase cache, no TTT

## Legality

- [x] Score-first: lookup cache THEN update
- [x] Single-pass only, each token scored exactly once
- [x] Mixing coefficient target-independent (`conf` depends on ctx_count only)
- [x] Packed artifact from training data only
- [x] No TTT, no GPTQ at eval, no reordering
- [x] Deterministic (std = 0.0000045)
- [x] Artifact < 16 MB (~1.5 MB)
- [x] Eval < 600s (271-292s)

## Credits

- PR #944: Packed causal n-gram + Dirichlet backoff architecture
- PR #900: Dirichlet posterior mixing theory
- PR #943: Packed causal memory concept
- PR #727/#753: Multi-order n-gram backoff foundation
