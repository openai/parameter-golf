# Record: 10L Int5-MLP3× + BigramHash(4096) + Int6 QAT + Sliding Eval

## Author
suchihype

## Summary
10-layer transformer with 3× MLP (int5 quantized), BigramHash(4096/128), full-run Int6 QAT with aligned STE, and sliding window evaluation (stride 64). Schedule optimized via Optuna TPE sweep on 8×H100.

## Architecture
- **Layers:** 10 (U-Net encoder-decoder with skip connections)
- **Model dim:** 512
- **Heads / KV heads:** 8 / 4 (GQA)
- **MLP multiplier:** 3.0× (hidden 1536), ReLU²
- **Vocab:** 1024 (tied embeddings)
- **BigramHash:** vocab 4096, dim 128
- **SmearGate:** per-dim learned gate blending current + previous token
- **RoPE:** base 50000, full dims (no partial RoPE)

## Training
- **Optimizer:** Muon (matrix params) + AdamW (embeddings, scalars)
- **Learning rates:** matrix=0.035, tied_embed=0.045, scalar=0.035
- **Muon weight decay:** 0.045 (decoupled)
- **Muon momentum:** 0.99, warmup from 0.92 over 1500 steps
- **Grad clip norm:** 0.35
- **Warmdown iters:** 2000
- **Warmup steps:** 20 (JIT warmup, state reset after)
- **Batch tokens:** 786,432
- **Sequence length:** 2048

## Quantization
- **Training:** Full-run Int6 QAT with STE (aligned with export: scale=amax/31, range [-32,31])
- **Export:** Mixed int5 (MLP) + int6 (attention) + FP16 (embeddings)
- **Compression:** zstd level 22

## Evaluation
- **Mode:** Sliding window, stride 64
- **Metric:** Tokenizer-agnostic bits per byte (FineWeb validation)

## Results (3-seed validation, 8×H100 SXM, 600s)

| Seed | Steps | Val BPB | Roundtrip BPB | Submission Bytes |
|------|-------|---------|---------------|-----------------|
| 1337 | 5,596 | 1.1646 | 1.1486 | 15,429,000 |
| 42 | 5,567 | 1.1654 | 1.1492 | 15,392,000 |
| 7 | 5,593 | 1.1649 | 1.1488 | 15,433,000 |
| **Mean** | **5,585** | **1.1650** | **1.1489** | — |
| **Std** | — | **0.0004** | **0.0003** | — |

## Key Insights
1. **Int6 STE must match export exactly** — training with scale=amax/127 but exporting with scale=amax/31 causes 0.18 bpb degradation
2. **EMA and SWA both hurt with full-run QAT** — weight averaging moves weights off the int6 grid, destroying quantization transparency
3. **Higher LRs are optimal for fewer steps** — 0.035/0.045 beat leaders' 0.02/0.03 at ~5500 steps on 8×H100
4. **Int5 MLP enables MLP 3×** — saves ~1.5MB, allowing bigger MLP within the 16MB cap
5. **Sliding eval gains ~0.023 bpb for free** — purely evaluation-side improvement
6. **Optuna TPE sweep found better schedule than hand-tuning** — lower embed LR and shorter warmdown

## Techniques Explored but Not Used
- **Late QAT:** Works on 1×H100 but unnecessary with full-run QAT on 8×H100
- **EMA/SWA:** Both degrade quantized quality with full-run QAT
- **Partial RoPE:** No improvement over full RoPE in our stack
- **XSA (cross-sequence attention):** Marginal gain (~0.0001), added variance
- **BigramHash 8192:** No improvement over 4096, eats size headroom
