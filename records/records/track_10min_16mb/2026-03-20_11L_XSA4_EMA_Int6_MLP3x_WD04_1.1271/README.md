## Record: 11L XSA + EMA + Int6 MLP3x + WD=0.04 (val_bpb: 1.1271)

**val_bpb = 1.1271** (sliding window, stride=64) | **15.5 MB** artifact | 8xH100 SXM, 600s

Previous: [PR #70](https://github.com/openai/parameter-golf/pull/70) (9L, 1.1659) → [PR #164](https://github.com/openai/parameter-golf/pull/164) (9L, 1.1524) → [PR #198](https://github.com/openai/parameter-golf/pull/198) (11L, 1.1318) → this

### Changes from PR #198

| | [PR #198](https://github.com/openai/parameter-golf/pull/198) | This |
|---|---|---|
| val_bpb (sliding s64) | 1.1318 | **1.1271** |
| XSA | None | Last 4 layers |
| Weight averaging | SWA (~8 checkpoints) | EMA (decay=0.997, every step) |
| Artifact | 15.7 MB | 15.5 MB |
| Everything else | Same | Same |

### What's new

1. **Exclusive Self Attention (XSA)** on last 4 layers. After the standard attention output, XSA subtracts the component aligned with each token's own value vector using an efficient GQA-aware reshape (no repeat_interleave). This encourages attention to capture only information orthogonal to what the token already knows, improving context modeling. Zero new parameters, ~2ms/step overhead.

2. **EMA replacing SWA**. Instead of collecting periodic SWA checkpoints during warmdown, we maintain an exponential moving average shadow model on GPU that updates every step: `ema = 0.997 * ema + 0.003 * param`. The EMA weights are used for quantization and eval. Smoother averaging than periodic SWA, better generalization and artifact compression.

### Carried from PR #198

- 11 transformer layers with U-Net skip connections
- Orthogonal + muP-scaled init on all large matrices
- 3x MLP (hidden=1536), relu² activation
- Int6 mixed quantization + zstd-22 (int6 on MLP+attention, int8 on embeddings)
- Weight decay 0.04 (Muon + AdamW)
- SmearGate (learned token blending gate, ~512 params)
- Bigram Hash Embedding (2048-bucket, dim=128, projected to 512)
- FlashAttention 3 (direct flash_attn_func calls)
- Sequence length 2048 with NTK-aware RoPE
- Muon optimizer, momentum 0.99 with warmup, warmdown 3000 iters, grad clip 0.3

### Configuration

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Key Metrics

- 7,103 steps in 600s (84ms/step)
- ~5.6B train tokens (7,103 steps x 786,432 tokens/step)
- Peak memory: ~20,400 MiB per GPU

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1427 |
| Int6 roundtrip val_bpb | 1.1494 |
| **Int6 sliding val_bpb (s64)** | **1.1271** |
| Compressed artifact (int6+zstd) | 15,468,512 bytes |
| Code size | 66,133 bytes |
| **Total submission size** | **15,534,645 bytes** |

### Reproducibility

| Seed | Steps | Sliding s64 | Artifact |
|------|-------|-------------|----------|
| **1337** | **7,103** | **1.1271** | **15,534,645** |
| 42 | 7,094 | 1.1286 | 15,745,973 |
| 2025 | 7,107 | 1.1284 | 15,649,516 |

Mean val_bpb: **1.1280**. Submitted: seed 1337 (best). Inter-seed variance: 0.0015.

### Included files

- `train_gpt.py` — full training + quantization + evaluation script
- `train.log` — training log from best seed (1337)
- `train_seed1337.log`, `train_seed42.log`, `train_seed2025.log` — all seed logs
- `submission.json` — leaderboard metadata
