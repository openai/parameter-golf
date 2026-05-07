# Record: 11L Parallel Muon + N-gram Backoff Cache — val_bpb 0.2841

**3-seed mean val_bpb: 0.2841** (std 0.0001) | **~15.92 MB** | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | EMA bpb | Quantized bpb | **N-gram bpb** |
|------|----------|-------|---------|---------------|----------------|
| 1337 | 88.6ms | 6,774 | 1.1193 | 1.1279 | **0.2841** |
| 42 | 88.6ms | 6,772 | 1.1194 | 1.1276 | **0.2840** |
| 2024 | 88.7ms | 6,769 | 1.1191 | 1.1279 | **0.2840** |
| **Mean** | **88.6ms** | **6,772** | **1.1193** | **1.1278** | **0.2841** |

## Key Innovation: N-gram Backoff Cache

Eval-time order 2-9 backward-looking N-gram cache with entropy-adaptive alpha blending:

```
for each 65K-token chunk:
    Phase 1 -- SCORE: sliding window (stride=64) with N-gram interpolation
        - For each token, blend model P(token) with N-gram P(token) using adaptive alpha
        - Alpha determined by model entropy and N-gram order (higher orders = higher weight)
    Phase 2 -- UPDATE: add scored tokens to N-gram frequency tables (backward-looking only)
```

N-gram cache reduces BPB by 4x (1.1278 -> 0.2841) by exploiting repeated phrases and patterns in the validation data. Score-first: cache only contains already-scored tokens.

- **4M hash buckets**, order 2-9 with XOR-of-products hashing
- **Entropy-adaptive alpha**: sigmoid(entropy_scale * (entropy - center)), scaled by per-order multipliers
- **Per-order multipliers**: orders 2-3 suppressed (0.3x), orders 5-9 boosted (2.0x)
- **65K-token chunks**: cache refreshes every 65K tokens for maximum coverage

## Architecture (26.8M params)

- 11L, 512d, 8H/4KV (GQA), MLP 3x LeakyReLU(0.5)²
- Parallel Muon with parameter banking + batched Newton-Schulz
- SmearGate, BigramHash(1024), Value Residual, Gated Attention
- XSA4, Partial RoPE(16/64), U-Net skips, OrthoInit
- EMA(0.997) + SWA, Late QAT, GPTQ-lite int6 + zstd-22
- Flash Attention 3, torch.compile(fullgraph=True)

## Timing

- Training: 600s (6,772 steps at 88.6ms/step)
- Eval (N-gram): ~420s
- Total: ~1020s (within 600s train + 600s eval budgets)

## Compliance

- [x] Training under 600s
- [x] Eval under 600s (N-gram ~420s)
- [x] Artifact under 16,000,000 bytes
- [x] N-gram cache is strictly backward-looking (updated AFTER scoring)
- [x] No training data access during evaluation
- [x] No oracle/hindsight selection

## Credits

- N-gram cache concept: PR #659 by @deanbrr, PR #674 by @newjordan
- Multi-order backoff + entropy-adaptive: PR #702 by @lukacf
- Fine-grained chunk updates: PR #843 by @quietsmile
- Parallel Muon / Parameter Banking: PR #399 by @abaybektursun
- LeakyReLU²: PR #493 by @parinzee
- Base model stack: PR #414 by @signalrush
