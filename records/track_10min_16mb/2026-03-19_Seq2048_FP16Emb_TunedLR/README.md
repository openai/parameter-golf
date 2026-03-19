# 10L Int6+Zstd MLP2.6x Muon0.99 Sliding Window

## Summary

Stacked improvements on the Naive Baseline:

1. **10 transformer layers** (from 9): Extra capacity.

2. **Full int6 quantization**: All 2D block weights quantized to [-31, 31] range (63 levels) instead of int8 [-127, 127]. Stored in int8 container but with much lower entropy.

3. **zstd-22 compression** (from zlib-9): Better compression for int6 data, saving ~4MB vs zlib. This freed space enables the wider MLP.

4. **MLP hidden 1344** (from 1024, 2.625x model_dim): Wider MLP significantly improves modeling capacity. 64-aligned for H100 matmul tiles.

5. **FP16 tied embedding passthrough**: Dual-purpose embedding kept in fp16 instead of int6 quantization.

6. **Sequence length 2048** (from 1024): Longer context per training step.

7. **Muon momentum 0.99** (from 0.95): Higher momentum with warmup from 0.92 over 1500 steps.

8. **Lower learning rates**: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.04.

9. **Gradient clipping**: GRAD_CLIP_NORM=0.3 (from 0.0). Stabilizes training with longer sequences.

10. **Sliding window evaluation** (stride=64): Overlapping windows give every scored token ~2000 tokens of context.

Warmdown=3600, warmup=20.

## Configuration

```bash
MLP_HIDDEN=1344 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Layout: `NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_HIDDEN=1344 TRAIN_SEQ_LEN=2048`

## Results

| Seed | Steps | val_bpb (standard) | val_bpb (sliding) | Artifact size |
|------|-------|--------------------|--------------------|---------------|
| 1337 | 8,692 | 1.1835 | 1.1625 | 15,678,040 |
| 42 | ~8,690 | ~1.1840 | 1.1639 | ~15,678,000 |
| 3 | ~8,690 | ~1.1835 | 1.1632 | ~15,678,000 |

**Mean val_bpb (sliding): 1.1632** (std: 0.00070)
**Mean val_loss (sliding): 1.9640** (std: 0.00118)

Statistical significance vs SOTA (1.2244 BPB / 2.0727 val_loss):
- Improvement: 0.1087 nats (threshold: 0.005)
- t-statistic: -152.3, df=2, p << 0.01

Hardware: 8xH100 80GB HBM3, PyTorch 2.8.0+cu128, ~69ms/step avg.
Sliding window eval time: ~363s. Requires `pip install zstandard`.

## Included Files

- `train_gpt.py` (modified training script)
- `train_seed1337.log`, `train_seed42.log`, `train_seed3.log`
- `submission.json`
