# PR 1130 — Kitchen Sink V2 Improved (resid_lambdas + bigger bigram/VE + FA3 + sliding eval)

**Author:** alejandro (Gusanidas)
**Branch date:** 2026-03-30
**Claimed BPB:** 1.1140 (12-seed mean, std 0.0005)
**Artifact size:** 15,884,408 bytes (worst-case, seed 777)
**Seeds:** 12 seeds: 2, 7, 22, 42, 77, 99, 222, 777, 1337, 2026, 2222, 9999
**Hardware:** 8×H100

## Files retrieved
- `README.md`
- `records__track_10min_16mb__2026-03-29_KitchenSinkV2__README.md`
- `records__track_10min_16mb__2026-03-29_KitchenSinkV2__train_gpt.py`
- `records__track_10min_16mb__2026-03-29_KitchenSinkV2__submission.json`

## Claimed changes (from README, verbatim)
"Built on PR #549 / KitchenSinkV2 with the following additions:

1. Split early/late LR banks — separate Muon and Adam optimizers for the first and second half of layers
2. MiLe margin loss — triangle-scheduled margin loss with gamma=0.75, clamp_min=0.2
3. Cache + backout residual — layer 7 output cached and mixed back via learnable gate
4. LeakyReLU(0.5)² activation in MLP
5. XSA on last 7 layers (up from default 4)
6. Coprime-stride multi-shard data loader (PR #726 / #1060 style)
7. Train-data GPTQ int6 calibration (PR #1060) — calibration uses training data within the training budget (14s reserved from 600s)
8. Residual lambdas — learnable per-sublayer residual scaling (init sqrt(1.1), 5x scalar LR, no weight decay)
9. Bigger bigram hash — 6144 buckets (up from 2048), reducing collision ratio
10. Bigger value embeddings — dim=196 on layers 5,9,10 (up from dim=128 on layers 9,10)
11. Flash Attention 3 via flash_attn_interface
12. Sliding window eval with stride=64"
