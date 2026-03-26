# F1 Legal LB Results — New SOTA

## Config: Legal LB Profile
- `MLP_ACT=leaky_relu_sq`, `MLP_LEAKY_SLOPE=0.5`
- `XSA_LAST_N=4`, `BIGRAM_VOCAB_SIZE=1536`
- `TTT_FREEZE_BLOCKS=0`, `TTT_GRAD_CLIP=0.8`
- `F1_CORR_RANK=0` (no correction head)
- `DISTILL_ENABLED=0` (no distillation)
- Script: `concepts/f1/run_legal_lb.sh`
- Training: `concepts/f1/train_gpt.py`

## 3-Seed Sweep

| Seed | Steps | val@4000 | post_ema | pre-TTT sliding | post-TTT | artifact |
|------|-------|----------|----------|----------------|----------|----------|
| 1337 | 6,919 | 1.2147 | 1.1379 | 1.1196 | **1.1195** | 15.90MB |
| 42 | 6,911 | 1.2146 | 1.1380 | 1.1199 | **1.1200** | 15.61MB |
| 2045 | 6,914 | 1.2140 | 1.1372 | 1.1191 | **1.1190** | 15.81MB |
| **Mean** | **6,915** | **1.2144** | **1.1377** | **1.1195** | **1.1195** | |

## vs Previous SOTA

| | PR #587 (old) | F1 Legal LB (new) | Delta |
|---|---|---|---|
| pre-TTT sliding (1337) | 1.1203 | **1.1196** | **-0.0007** |
| post-TTT (1337) | 1.1204 | **1.1195** | **-0.0009** |
| post-TTT mean (3-seed) | 1.1215 | pending | |

## Key Changes from PR #587
1. `leaky_relu_sq` activation (slope 0.5) — replaces standard relu_sq
2. `XSA_LAST_N=4` — reduced from 11 (all layers) to last 4 only
3. `TTT_FREEZE_BLOCKS=0` — unfreezes all blocks during TTT (was 2)
4. `BIGRAM_VOCAB_SIZE=1536` — reduced from 2048
5. `TTT_GRAD_CLIP=0.8` — tighter than default 1.0
