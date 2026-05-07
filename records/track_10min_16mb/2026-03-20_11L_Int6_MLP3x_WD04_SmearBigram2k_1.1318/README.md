## 11-Layer Int6 + WD=0.04 + SWA + FA3 (val_bpb: 1.1318)

**val_bpb = 1.1318** (sliding window, stride=64) | **15.7 MB** artifact | 8xH100 SXM, 600s

### Changes from PR #164

| | [PR #70](https://github.com/openai/parameter-golf/pull/70) | [PR #164](https://github.com/openai/parameter-golf/pull/164) | This |
|---|---|---|---|
| val_bpb (sliding) | 1.1659 (s256) | 1.1524 (s256) | **1.1318 (s64)** |
| Layers | 9 | 9 | 11 |
| Params | 21.8M | 22.4M | 26.8M |
| Artifact | 14.9 MB | 15.4 MB | 15.7 MB |
| Steps (8xH100, 600s) | 12,485 | 8,390 | 7,412 |
| Step time | 48ms | 68ms | 81ms |
| Train seq_len | 1024 | 2048 | 2048 |
| Weight decay | None | None | 0.04 (Muon + AdamW) |
| SWA | No | No | ~8 checkpoint avg |
| Eval stride | 256 | 256 | 64 |
| Bigram buckets | n/a | 4096 | 2048 |

### What's new

1. **11 transformer layers** (was 9). Two extra layers add 4.4M parameters. The main driver of the BPB gain. Fits under 16 MB thanks to int6 compression headroom.

2. **Weight decay 0.04**. Applied to both Muon (decoupled WD on matrix params) and AdamW (on embeddings/scalars). Shrinks weight magnitudes, improving int6 quantization tolerance and zstd compression ratio.

3. **Stochastic Weight Averaging**. Collects ~8 checkpoints during warmdown (when LR scale < 0.5, every 200 steps) and averages them before quantization.

4. **Sliding window stride=64** (was 256). Each scored token now has nearly full 2048-token context. ~0.002 BPB gain over stride=256.

5. **Bigram vocab 2048** (was 4096). Halved the bigram hash table to save ~300 KB artifact space with <0.001 BPB cost.

6. **Tuned LRs**. Slightly higher than PR #164: matrix_lr=0.025, scalar_lr=0.025, tied_embed_lr=0.035.

### Carried from PR #164

- Orthogonal + muP-scaled init on all large matrices
- 3x MLP (hidden=1536), relu² activation
- Int6 mixed quantization + zstd-22 (int6 on MLP+attention, int8 on embeddings)
- SmearGate (learned token blending gate, ~512 params)
- Bigram Hash Embedding (token-pair features into residual stream)
- FlashAttention 3 (direct `flash_attn_func` calls)
- Sequence length 2048 with NTK-aware RoPE
- Muon optimizer, momentum 0.99 with warmup, warmdown 3000 iters, grad clip 0.3

### Configuration

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Key Metrics

- 7,412 steps in 600s (81ms/step)
- ~5.5B train tokens (7,412 steps x 786,432 tokens/step)
- Peak memory: 19,710 MiB per GPU

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1432 |
| Int6 roundtrip val_bpb | 1.1543 |
| **Int6 sliding val_bpb (s64)** | **1.1318** |
| Compressed artifact (int6+zstd) | 15,624,965 bytes |
| Code size | 64,415 bytes |
| **Total submission size** | **15,689,380 bytes** |

### Reproducibility

| Seed | Steps | Sliding s64 | Artifact |
|------|-------|-------------|----------|
| **1337** | **7,412** | **1.1318** | **15,689,380** |
| 42 | 7,407 | 1.1335 | 15,695,752 |
| 2025 | 7,412 | 1.1324 | 15,689,877 |

Mean val_bpb: **1.1326**. Submitted: seed 1337 (best). Inter-seed variance: 0.0017.

### Included files

- `train_gpt.py` — full training + quantization + evaluation script
- `train.log` — training log from best seed (1337)
- `train_seed1337.log`, `train_seed42.log`, `train_seed2025.log` — all seed logs
- `submission.json` — leaderboard metadata
