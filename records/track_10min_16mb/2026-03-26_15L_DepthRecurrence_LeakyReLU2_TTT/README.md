# Record: 15L Depth Recurrence + LeakyReLU² + Cosine TTT (3-seed mean val_bpb=1.1093)

**3-seed mean val_bpb: 1.1093** (std 0.0045) | < 15.75 MB | 8xH100 SXM, 600s training + 543s eval

## Results (8xH100 SXM)

| Seed | Steps | Pre-TTT BPB | Post-TTT Sliding BPB | Artifact |
|------|-------|-------------|----------------------|----------|
| 42 | 5165 | 1.1608 | **1.1048** | 15.75 MB |
| 1337 | 5166 | 1.1608 | **1.1092** | 15.67 MB |
| 2025 | 5170 | 1.1601 | **1.1138** | 15.65 MB |
| **Mean ± Std** | | | **1.1093 ± 0.0045** | |

## Key Innovation: BI-Guided Depth Recurrence

We use **Block Influence (BI) scores** (ShortGPT, arXiv:2403.03853) to identify redundant layers, then apply **weight tying** (depth recurrence) to share a single block across 5 layer positions. This gives 15 effective transformer layers from only 11 unique parameter blocks — fitting the same 16MB artifact budget as an 11-layer model while gaining 4 extra layers of depth.

### Block Influence Analysis

We trained a 15-layer model and measured BI (angular distance between each layer's input/output). Layers 9–13 showed the lowest BI scores (0.10–0.16), indicating near-identity transformations:

| Layer | BI Score | Role |
|-------|----------|------|
| 0 | 0.459 | High impact (encoder) |
| 6–8 | 0.230–0.235 | High impact (encoder/decoder boundary) |
| **9–13** | **0.104–0.156** | **Low impact → tied** |
| 14 | 0.203 | Moderate (final decoder) |

### How Tying Works

Layers 9, 10, 11, 12, 13 share **one physical block** (same Q/K/V/MLP weights). Each layer still has independent block-level scalars (attn_scale, mlp_scale, resid_mix). The U-Net skip connections remain intact.

- 10 unique blocks + 1 shared×5 = 15 virtual layers, ~27M unique params
- Int6 + zstd-22 compression → ~15.7 MB (under 16 MB budget)

### Why It Works

At 15 layers, even "redundant" layers contribute via the residual stream — they act as lightweight refinement passes. Tying forces the shared block to learn a **general refinement operation** useful at multiple depths, while per-layer scalars handle depth-specific calibration.

## Other Techniques

- **LeakyReLU(0.5)²** — Preserves negative gradient flow through MLP. -0.003 BPB vs relu². Credit: PR #493, PR #518.
- **20-epoch Cosine TTT** — Full-weight AdamW test-time training with cosine LR + per-layer LR (3× for mlp.proj, 0.5× for mlp.fc). ~444s on 8xH100. Credit: PR #481.
- **Deduplication-aware export** — Tied weights are stored once with a reconstruction map, halving the quantized artifact for shared layers.

## Architecture

- 15L (10 unique + 1 shared×5), 512d, 8H/4KV GQA, MLP 3x
- LeakyReLU(0.5)², U-Net skip connections
- XSA last 4, Partial RoPE 16/64, LN Scale, VE128
- SmearGate + BigramHash(2048), OrthoInit
- EMA(0.997) + Tight SWA, Late QAT@0.15
- Int6 + GPTQ-lite + zstd-22
- FA3 (FlashAttention 3)

## Timing

| Phase | Time |
|-------|------|
| Training | 600s (5166 steps at 116ms/step) |
| Quantization + export | ~50s |
| 20-epoch cosine TTT | ~444s |
| Sliding window eval (stride=64) | ~99s |
| **Total eval** | **~543s (< 600s)** |

## Reproduce

```bash
pip install zstandard sentencepiece
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

SEED=1337 NUM_LAYERS=15 TIE_LAYERS=9,10,11,12,13 \
  TTT_EPOCHS=20 TTT_LR=0.0005 TTT_BATCH=32 \
  DIFF_ATTN=0 VRES_ENABLED=0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Base model**: signalrush (PR #374, PR #414) — XSA, Partial RoPE, LN Scale, EMA, GPTQ-lite
- **LeakyReLU²**: PR #493 (@parinzee), PR #518 (@sofiabod)
- **Cosine TTT**: PR #481 (@mrdavtan), PR #518 (@sofiabod)
- **Block Influence**: ShortGPT (arXiv:2403.03853)
- **SmearGate, BigramHash, OrthoInit**: PR #65 (@aquariouseworkman), PR #162 (@raahilshah)
