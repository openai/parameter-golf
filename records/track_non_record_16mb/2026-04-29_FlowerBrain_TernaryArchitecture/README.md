# Flower Brain: 6-Cell Ternary Architecture for Parameter Golf

**val_bpb = 1.1155** (pre-quant, 1xH100 SXM) | **10.4 MB** (35% under 16 MB budget)

**Category:** Unlimited Compute (1xH100 SXM, ~60 min training — NOT eligible for 10-min 8xH100 record track)
**Author:** G3sparky (Gavin Saunders)

---

## Summary

A novel ternary neural architecture where 6 specialized cells — arranged in a Flower of Life hexagonal topology — replace the standard monolithic transformer. Each cell uses BitLinear layers with ternary weights {-1, 0, +1} and Straight-Through Estimator (STE) quantization-aware training.

Pre-quant val_bpb of **1.1155** is competitive with main-leaderboard entries. Post-quant ternary gap (0.68 BPB) remains the key challenge. Experimental findings on the void fraction equilibrium and STE quantization dynamics are included.

---

## Results

| Metric | Scaled (H100) | Original (4060) |
|--------|--------------|-----------------|
| Pre-quant val_bpb | **1.1155** | 1.3610 |
| Post-quant val_bpb | 1.7996 | 1.7892 |
| Quantization gap | 0.68 BPB | 0.43 BPB |
| Submission size | **10.4 MB** | 5.85 MB |
| Parameters | 32.5M | 17.3M |
| Dimensions | 512-dim, 12 layers | 384-dim, 8 layers |
| Void fraction | 17.4% | 16.4% |
| Hardware | 1x H100 SXM 80GB | 1x RTX 4060 8GB |
| Training time | ~60 min | ~30 min |
| Throughput | 728K tok/s | 58K tok/s |

---

## Architecture

### 6-Cell Flower Brain

| Cell | Role |
|------|------|
| Embed | Token embedding + positional encoding |
| Attention | Multi-head self-attention (GQA 8/4) |
| Transform | MLP feed-forward (mult 3.0) with gated activation |
| Context | Cross-shape attention (XSA) for long-range context |
| Routing | Depth recurrence controller + skip gates |
| Prediction | Output projection + language modeling head |

### Key Design Choices

- **BitLinear layers:** Ternary weights {-1, 0, +1} with per-element STE. Threshold: sign(w) * (|w| > mean(|w|)).
- **512-dim model, 12 layers:** Scaled from original 384-dim/8-layer to improve capacity.
- **Depth recurrence:** Layers 3-5 re-executed (2 loops), providing 17 virtual layers from 12 physical layers.
- **Mixed compression:** Ternary packing (2 bits/weight) for MLP layers + int6 GPTQ for attention weights + brotli compression.

---

## Experimental Findings

Three training configurations tested on the same architecture:

| Config | Pre-quant | Post-quant | Gap | Void |
|--------|-----------|------------|-----|------|
| STE + standard WD=0.095 (Run 1) | **1.1155** | 1.7996 | 0.68 | 17.4% |
| STE + WD=0, LR=0.04 (Run 2) | 1.1266 | 3.7931 | **2.67** | 17.4% |
| fp16 no-STE baseline | 1.3824 | 1.6760 | **0.29** | 15.9% |

### Key Findings

1. **Void fraction is architecture-determined.** All three configs converge to 15.9-17.4% void regardless of training regime. The theoretical 30% equilibrium applies to different architectures.

2. **STE makes quantization gap worse, not better.** fp16 (no STE) has a 0.29 BPB gap; STE-trained has 0.68. The STE pushes weights into a distribution that ternary projection handles worse than natural fp16 weights.

3. **Weight decay regularizes for quantization.** Removing WD (Run 2) caused a catastrophic 2.67 BPB gap. WD keeps weights compact and ternary-friendly.

4. **Gap B is a projection problem, not a training problem.** The fix is in the ternary projection method, not in training hyperparameters.

---

## Compression

| Method | Size | BPB |
|--------|------|-----|
| Full precision (fp32) | ~130 MB | 1.1155 |
| Mixed ternary + GPTQ | **10.4 MB** | 1.7996 |
| Standard GPTQ int6 (baseline) | ~16 MB | ~1.12 |

Ternary weights at 2 bits/weight + brotli compression achieve 12x compression over fp32. The tradeoff is a 0.68 BPB gap — the key research frontier for ternary architectures.

---

## Relation to Competitive Submission

Our competitive submission (PR #1858, 0.9727 BPB with anti-hijack gate) uses the standard transformer architecture with score-first TTT + PPM-D byte mixture. This submission demonstrates the **Flower Brain ternary architecture** — our own novel design:

- The void compass diagnostic was born from Flower Brain void fraction monitoring
- The 16-17% void fraction equilibrium is a new finding about this architecture class
- The ternary {-1, 0, +1} weight structure is the same principle that produced 76.5% accuracy in ternary PNN vs 15.3% binary (p = 2.18e-11 across 50 seeds)

---

## Reproduction

```bash
# Single H100 (unlimited compute)
NUM_LAYERS=12 MLP_MULT=3.0 MAX_WALLCLOCK_SECONDS=3600 SEED=42 COMPRESSOR=brotli \
  python3 train_gpt_ternary.py

# Single RTX 4060 (original config)
MODEL_DIM=384 NUM_LAYERS=8 MAX_WALLCLOCK_SECONDS=1800 SEED=42 COMPRESSOR=brotli \
  python3 train_gpt_ternary.py
```

---

## Prior Work and Credits

- Parameter Golf baseline: openai/parameter-golf
- GPTQ: Frantar et al. (2022)
- BitLinear / Ternary QAT: inspired by BitNet b1.58 (Ma et al., 2024)
- Depth recurrence: competition community innovation
- Void fraction research: Saunders (2026), AU Patent 2026902541

---

*G3sparky — Gavin Saunders, April 2026*
