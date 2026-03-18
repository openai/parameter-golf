# Aweb Depth Recurrence

## Approach

**Core insight:** The baseline uses 9 unique transformer blocks at 512 dim, consuming only 10GB of 80GB available H100 VRAM. This leaves massive headroom for a deeper, wider model.

**Depth recurrence** (Universal Transformer, Dehghani et al. 2019; revisited at ICLR 2025-2026) decouples parameter count from computational depth by sharing weights across layers. Instead of paying the parameter cost for each layer independently, we create a small set of unique blocks and cycle through them multiple times.

## Architecture

| Property | Baseline | Ours | Advantage |
|----------|----------|------|-----------|
| Unique blocks | 9 | 4 | 2.25× fewer parameters per layer |
| Effective depth | 9 | 24 | 2.67× deeper computation |
| Model dimension | 512 | 768 | 1.5× wider representation |
| KV heads | 4 | 4 | Same GQA ratio |
| Attention heads | 8 | 8 | Head dim: 96 (vs 64) |
| MLP multiplier | 2× | 2× | Same |
| Vocab size | 1024 | 1024 | Same |
| Tied embeddings | Yes | Yes | Same |
| Parameter count | ~17M | ~17.3M | Similar budget |

## How It Works

```
Input tokens → Embedding → RMSNorm → x0

ENCODER (12 effective layers):
  for i in 0..11:
    block_idx = i % 4          # Cycle through 4 unique blocks
    x = blocks[block_idx](x, x0)
    skips.push(x)              # Store for U-Net skip connections

DECODER (12 effective layers):
  for i in 0..11:
    x = x + skip_weight[i] * skips.pop()   # U-Net skip
    block_idx = (12 + i) % 4               # Same 4 blocks
    x = blocks[block_idx](x, x0)

Output → RMSNorm → Tied embedding projection → Softcap → Loss
```

Each unique block sees the input 6 times during a forward pass, allowing it to iteratively refine the representation — similar to how diffusion models iteratively refine images. The skip connections (borrowed from the baseline's U-Net pattern) help gradient flow across the deep recurrent structure.

## Why This Should Work

1. **Scaling law insight:** This challenge optimizes L(N) — lowest loss for fixed parameter count N. Depth recurrence is the cleanest way to decouple depth from N, giving more compute per parameter.

2. **Empirical evidence:** The 4-hour non-record baseline (same 9-layer architecture, just longer training) reaches 1.1749 BPB pre-quantization. Our 24-layer model has more capacity per forward pass, so it should converge faster within the 10-minute budget.

3. **VRAM headroom:** The baseline uses 10GB of 80GB available VRAM. Going from 512→768 dim and 9→24 effective depth increases compute but stays well within H100 memory.

## Training Configuration

```bash
RUN_ID=aweb_depth_recurrence \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_UNIQUE_LAYERS=4 \
NUM_REPEATS=6 \
MODEL_DIM=768 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Quantization

Same int8 per-row quantization + zlib compression as baseline. The shared weights are only stored once (4 unique blocks), so the compressed artifact is actually smaller than the baseline despite the wider model.

## References

- Dehghani et al., "Universal Transformers" (ICLR 2019)
- "Revisiting the Shape Convention of Transformer Language Models" (2026)
- "Inner Thinking Transformer: Leveraging Dynamic Depth" (ACL 2025)
- Keller Jordan, modded-nanogpt (Muon optimizer)
