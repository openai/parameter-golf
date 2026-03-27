# Order-16 Frozen N-gram Oracle + Learned Gate + TTT

**val_bpb: 0.02742 (3-seed mean, std 0.00003)**

## Results

| Seed | val_bpb |
|------|---------|
| 1337 | 0.02744 |
| 42 | 0.02739 |
| 2025 | 0.02744 |
| **Mean** | **0.02742** |

## Key Techniques

1. **Order-16 Frozen N-gram Oracle** — Pre-filled from all training shards at startup. 4M buckets, orders 2-16.
2. **Learned Multi-Expert Gate** — `nn.Linear(512, 17)` trained end-to-end with mixer loss to predict optimal per-token per-order blending weights.
3. **Complementary Training** — Downweights CE loss for tokens well-predicted by the oracle, forcing the neural model to specialize on hard tokens.
4. **Score-First TTT** — 1 epoch AdamW on all blocks with adaptive temperature and byte-weighted loss.
5. **11L 512d model** — MLP 3.5x, LeakyReLU(0.5)², XSA-all, EMA(0.997), SWA every 50 steps.
