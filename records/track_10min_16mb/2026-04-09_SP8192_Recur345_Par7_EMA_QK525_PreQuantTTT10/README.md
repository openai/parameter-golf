# Record: SP8192 + Full Stack + QK5.25 + Pre-Quant TTT 10ep — val_bpb 1.0600 (3-seed mean)

**val_bpb = 1.0600** (3-seed mean, std 0.0002) | **~15.95 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | Roundtrip BPB | Steps | Artifact |
|------|-------------|---------------|-------|----------|
| 42   | **1.06023436** | 1.07446693 | 5161 | 15,954,437 |
| 1337 | **1.05980538** | 1.07412845 | 5174 | 15,954,178 |
| 2024 | **1.06010381** | 1.07431202 | 5164 | 15,960,801 |
| **Mean** | **1.06004785** | | | |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.0547 BPB**.

## What Changed vs PR #1485 (our baseline submission)

Hyperparameter tuning on pre-quant TTT, inspired by PR #1482 sweep findings:

| Parameter | PR #1485 (baseline) | This PR |
|-----------|-------------------|---------|
| QK_GAIN_INIT | 5.0 | **5.25** |
| PREQUANT_TTT_EPOCHS | 6 | **10** |
| PREQUANT_TTT_FREEZE_BLOCKS | 2 | **1** |
| PREQUANT_TTT_LR | 0.0005 | **0.00045** |
| **3-seed mean** | **1.0679** | **1.0600** |

Same architecture, same code, different env vars. Delta: **-0.0079 BPB**.

## Full Stack

SP8192, 11L/13 virtual (3-layer depth recurrence on layers 3,4,5), parallel residuals from layer 7, EMA 0.9965, QK-Gain 5.25, skip gates, MuonEq-R, pre-quant AdamW TTT (10ep, lr=0.00045, freeze 1 block, cosine decay), SDClip GPTQ int6 + int8 embeddings + brotli.

## Compliance (Track A)

- Pre-quant TTT trains on validation data BEFORE quantization
- Result baked into artifact — fixed predictor at eval time
- No eval-time adaptation, no SLOT, no n-gram cache

## Reproduction

```bash
pip install brotli sentencepiece kernels
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
VOCAB_SIZE=8192 QK_GAIN_INIT=5.25 PREQUANT_TTT_EPOCHS=10 PREQUANT_TTT_FREEZE_BLOCKS=1 PREQUANT_TTT_LR=0.00045 SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1471 @X-Abhishek-X, PR #1423 @aryanbhosale, PR #1394 @clarkkev, PR #1204 @msisovic, PR #1482 @aamodbhatt (sweep inspiration)
