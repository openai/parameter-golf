# GDN-Hybrid + TMA Megakernel + Brotli-11 (3-seed mean 1.01195 BPB)

- **3-seed mean:** **1.01195 BPB** (std 0.00061)
- **Best seed:** 1.01125 BPB (seed 42)
- **Artifact size range:** 15,765,907 to 15,804,631 bytes

## Per-seed results

| Seed | Steps | EMA BPB | Quantized BPB | Artifact bytes |
|------|------:|--------:|--------------:|---------------:|
| 42   | 2,212 | 1.000783 | 1.011245 | 15,765,907 |
| 1337 | 2,736 | 1.001806 | 1.012324 | 15,804,631 |
| 2024 | 2,758 | 1.000638 | 1.012281 | 15,802,165 |

## Architecture

- GDN-Hybrid: `[GDN x5] -> [SWA] -> [GDN x5] -> [SWA_shared]`
- GatedDeltaNet (FLA library), dim=512, 8 heads, head_dim=64, allow_neg_eigval=True
- Sliding Window Attention (weight-shared), window=512, QK-Gain 5.0
- Fused relu_sq MLP megakernel (Hopper TMA)
- SP1024 tokenizer, 33.5M parameters
- Fixed predictor — no TTT, no eval-time adaptation

## Training

- MuonEq-R (momentum=0.97) + AdamW
- EMA 0.997 + SWA
- TOTAL_ITERATIONS=2100, WARMDOWN_ITERS=1000
- GPTQ int6 + brotli quality 11

## Reproduction

```bash
pip install flash-linear-attention brotli zstandard sentencepiece
python3 data/cached_challenge_fineweb.py --variant sp1024

SEED=42 SCORE_MODE=1 ARCH_MODE=D \
TOTAL_ITERATIONS=2100 WARMDOWN_ITERS=1000 \
VAL_LOSS_EVERY=0 GPTQ_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt_gdn_mega.py
```
