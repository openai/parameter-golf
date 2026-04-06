# Record: SP8192 + Pre-Quant TTT + QK-Gain 5.0 — val_bpb 1.0791 (3-seed mean)

**val_bpb = 1.0791** (3-seed mean, std 0.0012) | **~15.12 MB** | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | **Sliding BPB** | Artifact |
|------|-----------------|----------|
| 42   | **1.0802**      | 15,123,918 |
| 314  | **1.0778**      | 15,118,254 |
| 999  | **1.0794**      | 15,127,567 |
| **Mean** | **1.0791** | |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.0356 BPB**.

## Key Change: QK-Gain 5.0 on the SP8192 + Pre-Quant TTT stack

Takes PR #1394 (@clarkkev) + PR #1364 pre-quant TTT and adds QK-Gain 5.0 (from 4.0). The base stack: SP8192, MLP 4x, depth recurrence (loop 4,5), MuonEq-R, SDClip, GPTQ embeddings, sigmoid-gated U-Net skips, brotli.

## Compliance (Track A — Fixed Predictor)

- No eval-time adaptation — model frozen after training + pre-quant TTT + GPTQ
- No SLOT, no n-gram cache
- Pre-quant TTT adapts EMA weights BEFORE GPTQ quantization (baked into artifact)
- Standard sliding-window eval (stride=64)
- All four conditions from Issue #1017 satisfied

## Reproduction

```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest
SEED=42 QK_GAIN_INIT=5.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1394 @clarkkev, PR #1364 @stukenov, PR #1416 @erichroepke, PR #1217 @bigbag, PR #1204 @msisovic, PR #1260 @dexhunter, PR #1019 @abaybektursun
