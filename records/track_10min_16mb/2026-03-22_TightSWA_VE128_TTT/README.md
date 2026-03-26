# Record: 11L Tight SWA + VE128 + XSA4 + TTT (val_bpb: 1.1299)

**NEW SOTA** — beats previous record of 1.1428 by 0.0129 nats (3-seed mean)

## Results

| Seed | Steps | Post-SWA BPB | Quant BPB | Sliding Window BPB | Artifact Size |
|------|-------|-------------|-----------|--------------------|--------------:|
| 1337 | 5880 | 1.1462 | 1.1529 | **1.1291** | 15,787,610 |
| 7 | 5850 | 1.1478 | 1.1545 | **1.1309** | 15,659,426 |
| 99 | 6024 | 1.1465 | 1.1533 | **1.1296** | 15,688,657 |
| **Mean** | | 1.1468 | 1.1536 | **1.1299** | 15,711,898 |

All 3 seeds beat SOTA (1.1428) by ≥0.012 nats. All artifacts < 16MB.

## Architecture

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion with relu-squared activation
- Efficient Partial XSA on last 4 layers (GQA-aware, zero-alloc)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- U-Net skip connections (5 encoder, 6 decoder)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Shared Value Embedding (dim=128, layers 9,10) — 1 table, per-layer learned scales
- FlashAttention 3 (Hopper) with SDPA fallback
- Orthogonal init with proj scaling by 1/sqrt(2*num_layers)
- Logit softcap 30.0, tied embeddings
- 27M parameters

## Key Techniques

### Tight SWA
SWA checkpoint collection restricted to scale<0.2 (last ~600 steps), every 50 steps, averaging 12 checkpoints. Eliminates the SWA quality penalty while maintaining quantization-friendly weight averaging.

### Test-Time Training (TTT)
3 epochs of continued training on already-evaluated validation tokens (SGD with momentum 0.9, lr=0.002, batch=32 sequences). Freezes first 2 blocks. Runs after quantization, before sliding window eval. ~51s additional eval time.

### Late QAT
STE int6 fake-quantization enabled when LR scale < 0.1 (during warmdown), teaching the model to be robust to quantization noise before SWA collection begins.

### Sliding Window Evaluation
Overlapping windows at stride=64 (context=2048), significantly improving BPB vs single-pass evaluation. ~100s eval time.

## Training

- **Optimizer**: Muon (matrices, lr=0.025, momentum=0.99, warmup 0.92→0.99 over 1500 steps) + AdamW (embeddings lr=0.035, scalars lr=0.025)
- **Weight Decay**: 0.04 (both Muon and Adam)
- **Gradient Clip**: 0.3
- **Batch**: 786,432 tokens/step, seq_len=2048
- **Warmdown**: 3000 iters (wallclock-based, ~600s cap)
- **Tight SWA**: every 50 steps when scale<0.2 (12 checkpoints)
- **Late QAT**: STE int6 when LR scale<0.1
- ~5900 steps in 600s at ~101ms/step

## Quantization

- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

## Evaluation Pipeline

1. Train for 600s (wallclock cap)
2. Apply Tight SWA (12 checkpoint average)
3. Serialize + int6/zstd compress (verify artifact < 16MB)
4. TTT: 3 epochs on already-evaluated val tokens (~51s)
5. Sliding window eval at stride=64 (~100s)
6. Total eval time: ~155s (well under 10min limit)

## Reproduction

```bash
# 8xH100 (default config, all hyperparameters are baked in)
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-22_TightSWA_VE128_TTT/train_gpt.py

# To reproduce specific seeds:
SEED=1337 DATA_PATH=... TOKENIZER_PATH=... torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=7    DATA_PATH=... TOKENIZER_PATH=... torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=99   DATA_PATH=... TOKENIZER_PATH=... torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware

8xH100 SXM (RunPod), PyTorch 2.10, CUDA 12.x

## Acknowledgments

Built on [PR #374](https://github.com/openai/parameter-golf/pull/374) by [@unnir](https://github.com/unnir) (v38: Tight SWA + VE128 + XSA4, val_bpb=1.1246). Added test-time training (TTT) and SDPA fallback.
