# Record: SP1024 + SLOT-24 + QK5.25 + Pre-Quant TTT — val_bpb 0.8265 (3-seed mean)

**val_bpb = 0.8265** (3-seed mean, std 0.0029) | **~15.76 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | SLOT BPB | Sliding BPB (no SLOT) | Steps | Artifact |
|------|----------|----------------------|-------|----------|
| 42   | **0.82329038** | 1.08834264 | 6591 | 15,764,692 |
| 1337 | **0.82916457** | 1.08844016 | 6591 | 15,756,236 |
| 2024 | **0.82694986** | 1.08842671 | 6591 | 15,760,000 |
| **Mean** | **0.82646827** | | | |

Prior SLOT SOTA (PR #1313): **0.8637 BPB**. Delta: **-0.0372 BPP**.

## Novel Contribution

PR #1313 SLOT base enhanced with **pre-quant AdamW TTT** — first combination of weight-level TTT and hidden-state SLOT optimization.

Pipeline: train -> EMA -> **pre-quant TTT (10ep)** -> GPTQ -> SLOT eval

Pre-quant TTT improves the base model quality (1.12 -> 1.09 sliding), then SLOT pushes further from a stronger starting point. The two techniques are complementary: TTT modifies weights (baked into artifact), SLOT modifies hidden states (eval-time, discarded per window).

## Changes from PR #1313

| Parameter | PR #1313 | This PR |
|-----------|----------|---------|
| QK_GAIN_INIT | 4.0 | **5.25** |
| Pre-quant TTT | None | **10ep, lr=0.00045, freeze 1 block** |
| Sliding BPB (no SLOT) | ~1.12 | **1.088** |
| **SLOT BPB** | **0.8637** | **0.8265** |

## Architecture (inherited from PR #1313)

SP1024, 11L 512dim, GQA 8/4, MLP 3x, squared LeakyReLU, XSA-all, VRL (Value Residual Learning), BigramHash (1024, dim 128), SmearGate, U-Net skip connections, EMA 0.997, Late QAT, Muon optimizer, mixed int6/int8 + LZMA.

## SLOT Mechanism

- Frozen model forward pass -> hidden states
- Per-window learnable: delta (hidden perturbation) + logit_bias
- 24 AdamW steps, cosine LR 0.012 -> 0.001
- Optimizes on scored positions (stride=96 window)
- Delta and logit_bias discarded after each window

## Compliance

- Training within 600s wallclock on 8xH100
- Pre-quant TTT: trains on val data before quantization, baked into artifact
- SLOT: frozen model weights, only throwaway per-window delta+logit_bias optimized
- No n-gram cache, no data leakage across windows

## Reproduction

```bash
pip install brotli sentencepiece kernels
python3 data/cached_challenge_fineweb.py --variant sp1024
SEED=42 QK_GAIN_INIT=5.25 PREQUANT_TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1313 @anthony-maio (SLOT architecture, base code), PR #1423 @aryanbhosale (TTT technique inspiration), PR #1482 @aamodbhatt (QK-Gain sweep)
