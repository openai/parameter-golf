# Record: 11-gram Eval Cache + Hedge Mixer (val_bpb: 0.8609)

## Architecture

- 11L × 512d transformer, 26.6M params, GQA (8 heads, 4 KV)
- XSA on all 11 layers, Gated Attention, Partial RoPE (16/64)
- Value Embedding (VE) dim=64 on layers 7-10
- Uniform Int6 quantization + zstd-22 compression
- Tied embeddings, EMA (0.997) + SWA (every 50)

## Key Techniques

### 11-gram Eval Cache (−0.284 bpb)
Multi-order n-gram cache (orders 2-11) with entropy-adaptive alpha mixing. Score-first, update-after protocol (legal per @valerio-oai, Issue #140). 4M buckets per order, uint32 count tables. Order-adaptive entropy gating determines mixing weight based on model confidence.

### Hedge Mixer
Online multiplicative-weights expert ensemble between neural and n-gram-enhanced predictions. Adapts weighting per-document based on which expert performs better. Beta=2.0 learning rate.

### No TTT
N-gram cache replaces TTT's benefit entirely. TTT disabled to maximize eval time budget for sliding window + n-gram scoring.

## Results

| Seed | val_bpb | Artifact | Steps | Eval Time |
|------|---------|----------|-------|-----------|
| 42   | 0.8600  | 15,341,541 | ~6500 | ~188s |
| 1337 | 0.8611  | 15,918,565 | ~6500 | ~188s |
| 2025 | 0.8616  | 15,790,804 | 6526 | 188s |
| **Mean** | **0.8609** | — | — | — |

## Ablation

| Config | val_bpb | Δ |
|--------|---------|---|
| Roundtrip (no n-gram, no sliding) | 1.1452 | baseline |
| + Sliding window (stride=64) + 11-gram + Hedge | 0.8609 | −0.284 |

## Training

- Base: PR #549 architecture with XSA, Gated Attention, VE
- Optimizer: Muon (matrices) + AdamW (scalars/embeddings)
- ~6500 steps in 600s wall clock on 8×H100 SXM (~92ms/step)
- Late QAT enabled at 15% remaining steps

## Credits

- PR #549 (abaybektursun): base architecture
- PR #727, #758: n-gram cache reference implementations
- PR #731: Hedge Mixer concept
- @valerio-oai: confirming n-gram cache legality (Issue #140)
