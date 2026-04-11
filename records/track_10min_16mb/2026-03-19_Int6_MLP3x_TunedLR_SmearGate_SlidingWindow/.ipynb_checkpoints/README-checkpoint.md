# Int6 MLP3x + Tuned LR + SmearGate + SlidingWindow

## Summary

Four orthogonal improvements stacked on the baseline 9-layer, 512-dim GPT:

1. **Int6 mixed quantization + zstd-22**: Per-row int6 quantization ([-32,31]) on MLP and attention weight matrices, fp16 passthrough for tied embeddings, compressed with zstd level 22. Saves ~4MB vs int8+zlib, enabling wider MLP.

2. **3x MLP expansion**: MLP hidden dimension 1536 (3x model_dim), up from baseline 1024 (2x). The freed budget from int6 quantization pays for the extra parameters. Provides ~0.019 BPB improvement.

3. **Tuned optimizer hyperparameters**: Halved learning rates (`matrix_lr=0.02`, `scalar_lr=0.02`, `tied_embed_lr=0.03`), higher Muon momentum (`0.99` with warmup from `0.92` over 1500 steps), extended warmdown (3000 iterations), gradient clipping (norm=1.0). These changes improve convergence within the 10-minute training budget.

4. **SmearGate**: A learned gate that blends each token's embedding with the previous token's embedding before the first transformer layer, providing bigram-like information at negligible parameter cost (~512 params).

Evaluation uses **sliding window** with stride=64: each token is scored with nearly full 1024-token context, improving BPB by ~0.03 over non-overlapping evaluation. Eval completes in ~64 seconds on 8xH100.

## Configuration

```
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_MULT=3 TIE_EMBEDDINGS=1
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=1.0
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024
EVAL_STRIDE=64
MAX_WALLCLOCK_SECONDS=600
```

## Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

- Training stopped at step **12308/20000** due to wallclock cap (600s, ~48.75ms/step)
- Model params: **21,778,504** (MLP 3x)
- Pre-quant: `val_loss:2.0060 val_bpb:1.1881`
- Int6+zstd roundtrip: `val_loss:2.0180 val_bpb:1.1952` (0.007 BPB quant degradation)
- **Sliding window (stride=64): `val_loss:1.9617 val_bpb:1.1618`** (eval time: 64s)
- Compressed model (int6+zstd): **15,084,480 bytes**
- Code: **59,656 bytes**
- **Total: 15,144,136 bytes** (under 16,000,000 limit by 855,864 bytes)
- Peak memory: 11,324 MiB allocated per GPU

## Approach Details

### Quantization Strategy

The key insight: use int6 (6-bit, 64 levels) for the large MLP and attention weight matrices, which are robust to quantization, while keeping the sensitive tied embedding in fp16. This mixed approach gives much better quality than uniform int6 while saving enough space to fit MLP 3x.

- MLP + attention 2D weights: int6 per-row quantization, scale in fp16
- Tied embedding (`tok_emb.weight`): kept as fp16 (no quantization)
- Small tensors (scales, norms, etc.): fp16 or fp32 passthrough
- Compression: zstd level 22 (better ratio than zlib-9 on int6 data)

### SmearGate

A simple pre-attention module that blends position `t`'s embedding with position `t-1`'s embedding via a learned sigmoid gate:
```python
g = sigmoid(self.gate)  # learned per-dim gate, initialized to 0
x = (1 - g) * x + g * x_prev  # x_prev is x shifted by 1 position
```
This provides each position with bigram information before the first attention layer sees it. Cost: 512 learnable parameters.

## Rejected Approaches

- **MoE (Mixture of Experts)**: 3.8x slower per step, fewer total steps, worse final BPB despite more parameters
- **Depth recurrence**: Large quantization gap (0.13 BPB) when weight-shared layers are quantized
- **Int8 + MLP 3x**: Doesn't fit in 16MB (20.5MB compressed)
- **Int7 quantization**: Better quality than int6 but compresses worse, only fits MLP 2.5x
- **Multi-token prediction**: Extra 512x512 projection pushes over size limit
- **Value embeddings**: No measurable improvement at this model scale
- **Lower logit softcap (15 vs 30)**: Slightly worse for this architecture
- **AdaGO optimizer**: Converges too slowly, worse than tuned Muon
