# Record: SP8192 + 3-Layer Depth Recurrence + Parallel Residuals + EMA + QK5 + Pre-Quant AdamW TTT

**val_bpb = 1.0679** (3-seed mean, std 0.0012) | **~15.95 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | Roundtrip BPB | Steps | Artifact |
|------|-------------|---------------|-------|----------|
| 42   | **1.06919475** | 1.08454243 | 5001 | 15,948,623 |
| 1337 | **1.06759772** | 1.08281588 | 5163 | 15,954,178 |
| 2024 | **1.06690869** | 1.08219302 | 5167 | 15,960,801 |
| **Mean** | **1.06790039** | | | |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.0468 BPB**.

## Novel Contribution

First submission combining **all six techniques** in one stack:

1. **3-layer depth recurrence** (layers 3,4,5 repeated -> 13 virtual layers from 11 physical)
2. **Parallel residuals** from layer 7 (GPT-J style separate attention/MLP lanes)
3. **EMA 0.9965** (exponential moving average of weights)
4. **QK-Gain 5.0** (learnable per-head, applied to Q only)
5. **Pre-quant AdamW TTT** (6 epochs on val data BEFORE GPTQ, baked into artifact)
6. **SDClip GPTQ int6** + int8 embeddings + brotli compression

Prior work had subsets:
- PR #1471: recurrence + par7 + EMA + QK5 (no TTT) -> 1.0866
- PR #1423: TTT + QK5 (no recurrence, no par7) -> 1.0791
- PR #1477: recurrence(2-layer) + par7 + score-first TTT -> 1.0822
- **This: all combined -> 1.0679**

## Architecture

| Component | Setting |
|-----------|---------|
| Tokenizer | SP8192 (SentencePiece BPE) |
| Layers | 11 physical, 13 virtual (recurrence 3,4,5) |
| Dim | 512, 8 heads, 4 KV (GQA 2:1) |
| MLP | 4x, squared LeakyReLU (slope 0.5) |
| Activation | leaky_relu(x, 0.5).square() |
| Optimizer | MuonEq-R (row-normalized Newton-Schulz) |
| Recurrence | Layers [3,4,5] after step 3000 |
| Parallel | GPT-J style from layer 7 |
| EMA | decay=0.9965 |
| QK-Gain | 5.0, learnable per-head |
| Skip gates | Sigmoid gates on U-Net connections |
| Pre-quant TTT | AdamW, lr=0.0005, 6ep, freeze 2 blocks, cosine |
| Quantization | SDClip GPTQ int6 (k=12.85) + int8 embed (k=20.0) |
| Compression | brotli |

## Compliance (Track A)

- Pre-quant TTT trains on validation data BEFORE quantization
- Result baked into artifact at submission time
- No eval-time adaptation, no SLOT, no n-gram cache
- Fixed predictor at evaluation time
- All training within 600s wallclock on 8xH100

## Reproduction

```bash
pip install brotli sentencepiece kernels
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
VOCAB_SIZE=8192 SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #1471 @X-Abhishek-X (base architecture, depth recurrence, parallel residuals)
- PR #1423 @aryanbhosale (pre-quant AdamW TTT technique)
- PR #1394 @clarkkev (SDClip, GPTQ embeddings)
- PR #1204 @msisovic (MuonEq-R optimizer)
