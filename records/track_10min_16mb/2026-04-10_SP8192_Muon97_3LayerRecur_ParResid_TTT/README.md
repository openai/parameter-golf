# Record: SP8192 + Muon 0.97 + 3-Layer Recurrence + Parallel Residuals + TTT — val_bpb 1.0802 (3-seed mean)

**val_bpb = 1.0802** (3-seed mean, std 0.0007) | 8xH100 SXM

## 3-Seed Results

| Seed | **TTT BPB** |
|------|-------------|
| 42   | **1.0795**  |
| 314  | **1.0808**  |
| 999  | **1.0804**  |
| **Mean** | **1.0802** |

Merged SOTA (PR #1493): **1.0810 BPB**. Delta: **-0.0008 BPB**.

## Key Change: Muon Momentum 0.97

Single hyperparameter change on the merged #1 stack (PR #1493): Muon momentum from 0.99 to 0.97. Validated by PR #1514 (@dexhunter) which showed 0.97 improves over 0.99 on the SP8192 base.

## Full Stack

PR #1493 base: SP8192, MLP 4x, 3-layer depth recurrence (L3-5), parallel residuals (L7+), QK-Gain 5.25, MuonEq-R, WD=0.095, EMA=0.9965, warmdown=0.72, SDClip, GPTQ embeddings, score-first TTT (3 epochs), brotli. Plus: **Muon momentum 0.97**.

## Compliance (Track B)

Score-first TTT (PR #461 framework). No SLOT, no pre-quant TTT, no n-gram cache. All four conditions from Issue #1017 satisfied.

## Reproduction

```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest
SEED=42 TTT_ENABLED=1 MUON_MOMENTUM=0.97 QK_GAIN_INIT=5.25 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1493 @bigbag (merged #1 base), PR #1514 @dexhunter (Muon 0.97), PR #1394 @clarkkev (SP8192), PR #1204 @msisovic (parallel residuals)
