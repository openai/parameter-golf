# Non-Record: Crawler Transformer 3f+2cx2 + SP8192 + SDClip + Post-Quant TTT — val_bpb 1.1372

**val_bpb: 1.1372** (3-seed mean, std 0.0004) | **~15.03 MB** | 1x RTX 6000 Ada, 18000s

### 3-Seed Results — d=736 int6 (1x RTX 6000 Ada 48GB)

| Seed | Steps | Pre-quant BPB | GPTQ Roundtrip BPB | **TTT BPB** | Artifact |
|------|-------|---------------|---------------------|------------|----------|
| 1337 | 6042 | 1.1232 | 1.1738 | **1.1372** | 15,025,540 B |
| 42 | 6012 | 1.1235 | 1.1732 | **1.1376** | 15,021,049 B |
| 2024 | 5977 | 1.1222 | 1.1746 | **1.1368** | 15,075,112 B |
| **Mean** | | **1.1230** | **1.1739** | **1.1372 (std 0.0004)** | |

Note: Trained on 1x RTX 6000 Ada for 5hr per seed (~6000 steps). Equivalent step count verified on 8xH100 cluster (6374 steps in 600s, SWA 1.1200).

### Additional config — d=832 mixed-int (int5 flat attention, MLP; int6 crawler attention, MLP, int8 embedding)

| Metric | d=736 int6 | **d=832 int5-flat** |
|--------|-----------|-------------------|
| Params | 38.3M | **47.4M** |
| Bits/weight | 3.13 | **2.67** |
| Pre-quant SWA | 1.1232 | **1.1092** |
| GPTQ Roundtrip | 1.1738 | 1.1851 |
| **TTT BPB** | **1.1372** | **1.1372** |
| Artifact | 15.03 MB | 15.86 MB |

Both configs converge to the same TTT score despite very different approaches. d=832 achieves **1.1092 pre-quant** (only 0.022 behind SOTA) with just 5 blocks, demonstrating the crawler architecture's learning efficiency. The higher quantization penalty (+0.076 vs +0.051) is fully recovered by TTT.

### Changes from PR #927

This builds on our previous submission PR #927 (Recursive Transformer 4B/7L, val_bpb 1.1696). Key changes:

| Change | PR #927 | This |
|--------|---------|------|
| Architecture | Recursive 4B/7L (d=1024) | **Crawler 3f+2cx2 (d=736/832)** |
| Tokenizer | SP1024 | **SP8192** |
| Quantization | Percentile search + LZMA | **SDClip + GPTQ + Brotli** |
| TTT freeze | freeze=5 (surgical) | **freeze=1** |
| Warmdown | 2000 steps fixed | **60% fraction** |
| Weight decay | 0.04 | **0.085** |
| val_bpb | 1.1696 | **1.1372** |

### Architecture: Crawler Transformer

Unlike the standard depth-recurrence approach (11L with shared layers), we use a **Crawler Transformer** architecture (inspired by @newjordan's crawler work):

- **3 flat blocks + 2 crawler blocks x 2 loops = 7 effective depth**
- Flat blocks: unique parameters, encoder-decoder with skip connections
- Crawler blocks: shared parameters, looped through the middle of the network
- dim=736 (int6) or dim=832 (mixed int5/int6), 16 heads (8 KV), MLP 4x, GQA
- BigramHash embedding (10240 buckets, 128 dim)
- SmearGate, ValueEmbedding (last 2 layers)
- XSA on all 7 layers
- **38.3M params (d=736) / 47.4M params (d=832)**

### Quantization

1. **QAT** (int6 fake-quantize via STE) from step 0
2. **SDClip**: `clip = k * std(row)` — k=12.85 for int6 blocks, k=20.0 for int8 embeddings
3. **Full Hessian GPTQ** with Cholesky error compensation, training-data calibration
4. **Brotli compression** (quality=11)
5. **No pruning** — all artifacts fit under 16MB natively
6. **Mixed-int option**: int5 flat blocks + int6 crawler blocks for d=832 (inspired by @newjordan's Midnight 12L mixed-int approach, PR #1458)

### Training

- Muon optimizer (momentum=0.99, WD=0.085) + Adam for scalars
- Warmdown fraction: 60% (linear)
- QK-Gain: 1.5, logit softcap: 30.0
- train_batch_tokens: 524,288, seq_len: 2048
- SP8192 tokenizer

### Test-Time Training (TTT)

- Sliding window with stride=64, chunk_tokens=32768
- SGD (lr=0.002, momentum=0.9), 3 epochs per chunk
- **freeze=1**: freezes first flat block + first crawler block
- Recovery: 0.037 BPB from GPTQ roundtrip (1.1739 -> 1.1372)
- Total penalty from pre-quant: only +0.014 BPB

### Credits

- **SDClip + SP8192 + GPTQ embeddings + Brotli**: PR #1394 by @clarkkev
- **XSA (extended self-attention)**: PR #549 by @abaybektursun
- **Sliding window TTT**: PR #549 by @abaybektursun
- **Crawler Transformer architecture**: inspired by @newjordan
- **Mixed-int quantization (int5 attn, int6 crawler)**: inspired by @newjordan PR #1458

### Run Command

```bash
# d=736 (int6 all blocks)
VOCAB_SIZE=8192 DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
MODEL_DIM=736 MAX_WALLCLOCK_SECONDS=18000 SEED=1337 \
RUN_ID=seed1337 \
python train_gpt.py

# d=832 (int5 flat, int6 crawler)
VOCAB_SIZE=8192 DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
MODEL_DIM=832 QAT_FLAT_BITS=5 MAX_WALLCLOCK_SECONDS=18000 SEED=1337 \
RUN_ID=seed1337_d832 \
python train_gpt.py
```
