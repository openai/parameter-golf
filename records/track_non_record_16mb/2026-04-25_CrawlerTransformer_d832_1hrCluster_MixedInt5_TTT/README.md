# Non-Record: Crawler Transformer 3f+2cx2 d=832 — Mixed Int5 GPTQ + Post-Quant TTT — val_bpb 1.0903

**val_bpb: 1.0903** | **15.96 MB** | 1x RTX 6000 Ada 48GB, 30 hours (1-hour 8xH100 cluster equivalent)

### Result Summary

| Stage | val_loss | val_bpb |
|-------|----------|---------|
| Pre-quant SWA | 2.7592 | **1.0684** |
| int8+SDClip roundtrip | 2.9393 | 1.1381 |
| GPTQ mixed-int (int5 flat-attn / int6 rest) roundtrip | 2.9090 | 1.1264 |
| **Post-quant TTT (freeze=1) on GPTQ artifact** | **2.8157** | **1.0903** |

- **Steps**: 30,374 / 50,000 (stopped by 30-hour wallclock cap)
- **Artifact**: 15,867,420 bytes (15.96 MB), zero pruning needed
- **Code**: 91,686 bytes
- **Total**: 15,959,106 bytes (under 16 MB budget)

### Comparison to 10-min Track Submission (PR #1579)

| Config | Steps | Pre-quant | TTT BPB | Hardware (effective) |
|--------|-------|-----------|---------|----------------------|
| d=736 int6 (10-min, PR #1579) | 6,042 | 1.1232 | 1.1372 | 10-min cluster |
| **d=832 int5-flat (1-hour, this)** | **30,374** | **1.0684** | **1.0903** | **1-hour cluster** |

6x training compute → -0.047 BPB improvement. Pre-quant alone (1.0684) already beats SOTA #1's TTT score (1.0808 from PR #1487).

### Architecture: Crawler Transformer

- **3 flat blocks + 2 crawler blocks × 2 loops = 7 effective depth**
- Flat blocks: unique parameters with skip connections
- Crawler blocks: shared parameters, looped through the network
- dim=832, 16 heads (8 KV), MLP 4x, GQA
- BigramHash, SmearGate, ValueEmbedding (last 2 layers), XSA on all 7 layers
- **47.4M parameters**
- SP8192 tokenizer (from `kevclark/parameter-golf` HuggingFace)

### Quantization Pipeline (Mixed Int5/Int6)

- **int5 (clip=15)** for flat-block attention only (12 matrices: c_q, c_k, c_v, attn.proj × 3 flat blocks)
- **int6 (clip=31)** for everything else (22 matrices: flat MLPs + all crawler blocks)
- **int8** for embeddings
- **SDClip** scale selection (k=12.85 blocks, k=20.0 embed) — from PR #1394
- **Full Hessian GPTQ** with Cholesky error compensation, training-data calibration
- **Brotli** compression (quality=11)
- **Zero pruning** — fits naturally at 15.96 MB

### Training Recipe

- 30-hour local run (1x RTX 6000 Ada 48GB) ≈ 1-hour 8xH100 SXM cluster
- Standard QAT int6 throughout training (no QAT int5 — that didn't help in earlier tests)
- Muon optimizer (momentum=0.99, WD=0.085) + Adam for scalars
- Warmdown fraction: 60% (linear)
- QK-Gain: 1.5, logit softcap: 30.0
- train_batch_tokens: 524,288, seq_len: 2048
- 30,374 steps in 30 hours (~3.55s/step on single GPU)

### Test-Time Training (TTT)

- Sliding window with stride=64, chunk_tokens=32768
- SGD (lr=0.002, momentum=0.9), 3 epochs per chunk
- **freeze=1**: freezes first flat block + first crawler block
- Recovery: 0.036 BPB from GPTQ roundtrip (1.1264 → 1.0903)
- Total penalty from pre-quant: only +0.022 BPB

### Key Learnings

1. **Mixed-int beats pruning**: At d=832, standard int6 needs 13.5% pruning (roundtrip 1.1664). Mixed int5 flat-attn / int6 rest fits naturally with no pruning (roundtrip 1.1264) — better quality at same artifact size.
2. **Int5 attention is robust, int5 MLP is not**: Quantizing only flat attention to int5 saves space without significant quality loss. MLP at int5 hurts much more.
3. **Pre-quant matters most**: 6x more training compute (1-hour cluster vs 10-min) gave 0.041 BPB improvement at the SWA stage, which carried through quantization and TTT.

### Credits

- **Crawler Transformer architecture**: inspired by @newjordan's crawler research (PR #1535)
- **Mixed-int quantization (int5 attn / int6 MLP)**: inspired by @newjordan's Midnight 12L (PR #1458)

### Run Command

```bash
# Training (30 hours local ≈ 1 hour 8xH100 cluster)
VOCAB_SIZE=8192 DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
MODEL_DIM=832 \
MAX_WALLCLOCK_SECONDS=108000 ITERATIONS=50000 \
SEED=1337 RUN_ID=d832_30hr \
python train_gpt.py
```

After training, requantize with int5 flat-attn + int6 rest, then run post-quant TTT.
