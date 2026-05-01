# Non-Record: Crawler Transformer 3f+2cx2 d=832 — Mixed Int5 GPTQ + Post-Quant TTT — val_bpb 1.0910 (4-hour cluster)

**val_bpb: 1.0910** | **14.71 MB** | 1x RTX 6000 Ada 48GB, 120 hours (4-hour 8xH100 cluster equivalent)

### Result Summary

| Stage | val_loss | val_bpb |
|-------|----------|---------|
| Pre-quant (last step, 122832/200000) | 2.7185 | **1.0526** |
| Pre-quant SWA (288 checkpoints) | 2.7452 | 1.0629 |
| int8+SDClip+zlib roundtrip | 2.9610 | 1.1465 |
| GPTQ mixed-int (int5 flat-attn / int6 rest) roundtrip | 2.9025 | 1.1238 |
| **Post-quant TTT (freeze=1) on GPTQ artifact** | **2.8176** | **1.0910** |

- **Steps**: 122,832 / 200,000 (stopped by 120-hour wallclock cap)
- **Artifact**: 14,622,509 bytes (14.62 MB), zero pruning needed
- **Code**: 92,135 bytes
- **Total**: 14,714,644 bytes (under 16 MB budget, ~1.25 MB headroom)

### Comparison to 1-hour Cluster Submission (PR #1817)

| Config | Steps | Pre-quant SWA | GPTQ roundtrip | TTT BPB | Artifact | Hardware (effective) |
|--------|-------|---------------|----------------|---------|----------|----------------------|
| d=832 int5-flat (1-hour, PR #1817) | 30,374 | 1.0684 | 1.1264 | **1.0903** | 15.96 MB | 1-hour cluster |
| **d=832 int5-flat (4-hour, this)** | **122,832** | **1.0629** | **1.1238** | 1.0910 | **14.62 MB** | **4-hour cluster** |

**Key finding**: 4x more training compute → marginal improvement at every stage *except* final TTT, which is essentially flat (+0.0007 BPB). The longer-trained model has wider weight distributions (larger int8 penalty) but compresses ~8% better with Brotli, freeing 1.25 MB of budget headroom.

### Why Compute Doesn't Help Past ~1 Hour Cluster Equivalent

1. **Multi-epoch saturation**: 122,832 steps × 524,288 tokens/step = ~64.4B tokens trained. At 10B tokens/epoch on FineWeb10B, the model passes through the dataset ~6.4 times. After epoch 1, marginal data gain drops sharply.

2. **Late warmdown timing**: `WARMDOWN_FRAC=0.6` starts the LR cooldown at step ~80k. By then the model is already memorizing rather than learning new patterns. Earlier warmdown (e.g., `WARMDOWN_FRAC=0.3-0.4`) would likely capture more of the saturation gain.

3. **TTT recovery ceiling**: GPTQ → TTT recovers 0.033 BPB (1.1238 → 1.0910), nearly identical to the 30hr run's 0.036 BPB recovery. The sliding TTT mechanism appears to have a fixed recovery budget that doesn't scale with model quality.

### Architecture: Crawler Transformer

Identical to PR #1817:
- **3 flat blocks + 2 crawler blocks × 2 loops = 7 effective depth**
- Flat blocks: unique parameters with skip connections
- Crawler blocks: shared parameters, looped through the network
- dim=832, 16 heads (8 KV), MLP 4x, GQA
- BigramHash, SmearGate, ValueEmbedding (last 2 layers), XSA on all 7 layers
- **47.4M parameters**
- SP8192 tokenizer (from `kevclark/parameter-golf` HuggingFace)

### Quantization Pipeline (Mixed Int5/Int6)

Identical to PR #1817:
- **int5 (clip=15)** for flat-block attention only (12 matrices: c_q, c_k, c_v, attn.proj × 3 flat blocks)
- **int6 (clip=31)** for everything else (22 matrices: flat MLPs + all crawler blocks)
- **int8** for embeddings
- **SDClip** scale selection (k=12.85 blocks, k=20.0 embed)
- **Full Hessian GPTQ** with Cholesky error compensation, training-data calibration
- **Brotli** compression (quality=11)
- **Zero pruning** — fits naturally at 14.62 MB (1.25 MB headroom available)

### Training Recipe

- 120-hour local run (1x RTX 6000 Ada 48GB) ≈ 4-hour 8xH100 SXM cluster
- Standard QAT int6 throughout training
- Muon optimizer (momentum=0.99, WD=0.085) + Adam for scalars
- Warmdown fraction: 60% (linear)
- QK-Gain: 1.5, logit softcap: 30.0
- train_batch_tokens: 524,288, seq_len: 2048
- 122,832 steps in 120 hours (~3.52s/step on single GPU)

### Test-Time Training (TTT)

Identical setup to PR #1817:
- Sliding window with stride=64, chunk_tokens=32768
- SGD (lr=0.002, momentum=0.9), 3 epochs per chunk
- **freeze=1**: freezes first flat block + first crawler block
- Recovery: 0.033 BPB from GPTQ roundtrip (1.1238 → 1.0910)

### Takeaways for Future Runs

1. **The 1-hour cluster is the right operating point** for d=832 + this recipe. More compute on the same recipe produces flat results.
2. **Budget headroom unlocks bigger models**: 1.25 MB of unused budget at d=832 + 4hr could afford d=896 or higher-precision embeddings.
3. **Warmdown_frac needs to scale inversely with epoch count**: long runs over-saturated data need earlier cooldown.
4. **TTT is the bottleneck**, not training**: Future improvement should target the post-quant TTT mechanism itself, not pre-quant model quality.

### Credits

- **Crawler Transformer architecture**: inspired by @newjordan's crawler research (PR #1535)
- **Mixed-int quantization (int5 attn / int6 MLP)**: inspired by @newjordan's Midnight 12L (PR #1458)

### Run Command

```bash
# Training (120 hours local ≈ 4 hours 8xH100 cluster)
VOCAB_SIZE=8192 DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
MODEL_DIM=832 \
MAX_WALLCLOCK_SECONDS=14400 ITERATIONS=200000 \
SEED=1337 RUN_ID=d832_4hr \
python train_gpt.py
```

After training, requantize with int5 flat-attn + int6 rest, then run post-quant TTT.
