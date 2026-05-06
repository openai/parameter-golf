# 10L Int6 QAT + Zstd MLP2.6x Muon0.99 Sliding Window

## Summary

Stacked improvements on the Naive Baseline:

1. **10 transformer layers** (from 9).

2. **STE int6 QAT**: Straight-through estimator fake quantization during training. Each CastedLinear forward pass applies `fake_quantize_int6(w)` — quantize to [-31,31], dequantize, with gradients flowing through via STE. This teaches the model to be robust to int6 quantization, **completely eliminating the quant gap** (pre-quant = post-quant loss).

3. **Full int6 quantization**: All 2D block weights quantized to [-31,31] (63 levels) in int8 container.

4. **zstd-22 compression**: Better than zlib for int6 data.

5. **MLP hidden 1344** (2.625x model_dim): Wider MLP enabled by int6+zstd savings.

6. **FP16 tied embedding passthrough**.

7. **Sequence length 2048**.

8. **Muon momentum 0.99**, warmup from 0.92 over 1500 steps.

9. **MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.04**.

10. **Gradient clipping** GRAD_CLIP_NORM=0.3.

11. **Sliding window evaluation** stride=64.

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

Requires: `pip install zstandard`

## Results

| Seed | Steps | val_bpb (standard) | val_bpb (sliding) | Artifact size |
|------|-------|--------------------|--------------------|---------------|
| 1337 | 8,319 | 1.1821 | 1.1610 | 15,558,319 |
| 42 | ~8,300 | ~1.1815 | 1.1598 | ~15,558,000 |
| 3 | ~8,300 | ~1.1810 | 1.1586 | ~15,558,000 |

**Mean val_bpb (sliding): 1.1598** (std: 0.00120)
**Mean val_loss (sliding): 1.9583** (std: 0.00203)

Quant gap: **0.0000** — STE QAT completely eliminated quantization loss.

Statistical significance vs SOTA (1.2244 BPB / 2.0727 val_loss):
- Improvement: 0.1144 nats (threshold: 0.005)
- t-statistic: -93.6, df=2, p << 0.01

Hardware: 8xH100 80GB HBM3, PyTorch 2.8.0+cu128, ~72ms/step avg.
QAT overhead: ~28% (72ms vs 69ms without QAT).
Sliding window eval time: ~370s.

## Included Files

- `train_gpt.py` (modified training script)
- `train_seed1337.log`, `train_seed42.log`, `train_seed3.log`
- `submission.json`
