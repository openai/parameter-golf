## Record: 11L Partial RoPE + LN Scale + EMA + XSA4 (val_bpb: 1.1248)

**val_bpb = 1.1248** (sliding window, stride=64) | **15.6 MB** artifact | 8xH100 SXM, 600s

Previous: [PR #70](https://github.com/openai/parameter-golf/pull/70) (9L, 1.1659) → [PR #164](https://github.com/openai/parameter-golf/pull/164) (9L, 1.1524) → [PR #198](https://github.com/openai/parameter-golf/pull/198) (11L, 1.1318) → [PR #287](https://github.com/openai/parameter-golf/pull/287) (11L, 1.1271) → this

### Changes from PR #287

| | [PR #287](https://github.com/openai/parameter-golf/pull/287) | This |
|---|---|---|
| val_bpb (sliding s64) | 1.1271 | **1.1248** |
| Partial RoPE | None (full 64d) | 16 of 64 dims |
| LN Scale | None | 1/sqrt(layer_idx+1) |
| Artifact | 15.5 MB | 15.6 MB |
| Everything else | Same | Same |

### What's new

1. **Partial RoPE (16 of 64 dims)**. Rotary position embeddings applied to only the first 16 of 64 head dimensions (25%). The remaining 48 dims attend without positional bias, allowing the model to learn position-invariant patterns. Zero new parameters.

2. **LN Scale**. RMSNorm outputs are scaled by 1/sqrt(layer_idx+1), damping deeper layers' contributions. Stabilizes training and improves convergence in deep models. Zero new parameters.

### Carried from PR #287

- 11 transformer layers with U-Net skip connections
- Exclusive Self Attention (XSA) on last 4 layers
- EMA weight averaging (decay=0.997, every step)
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
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Key Metrics

- 7,051 steps in 600s (85ms/step)
- ~5.5B train tokens (7,051 steps x 786,432 tokens/step)
- Peak memory: ~20,600 MiB per GPU

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1418 |
| Int6 roundtrip val_bpb | 1.1485 |
| **Int6 sliding val_bpb (s64)** | **1.1248** |
| Compressed artifact (int6+zstd) | 15,544,691 bytes |
| Code size | 67,617 bytes |
| **Total submission size** | **15,612,308 bytes** |

### Reproducibility

| Seed | Steps | Sliding s64 | Artifact |
|------|-------|-------------|----------|
| **2025** | **7,051** | **1.1248** | **15,612,308** |
| 42 | 7,061 | 1.1250 | 15,528,666 |
| 1337 | 7,063 | 1.1253 | 15,639,340 |

Mean val_bpb: **1.1250**. Submitted: seed 2025 (best). Inter-seed variance: 0.0005.

### Included files

- `train_gpt.py` — full training + quantization + evaluation script
- `train.log` — training log from best seed (2025)
- `train_seed2025.log`, `train_seed42.log`, `train_seed1337.log` — all seed logs
- `submission.json` — leaderboard metadata

### Note on Late QAT

The submitted code includes a Late QAT flag (`LATE_QAT=1`) intended to enable STE int6 fake-quantization in the final 4% of training. Post-submission analysis (credit: @152334H) revealed that `torch.compile` constant-folds the `CastedLinear._qat_enabled` class attribute at first trace, so the STE branch is dead-code-eliminated and never activates during training. Late QAT had no effect on the results. The score is driven entirely by Partial RoPE and LN Scale.
